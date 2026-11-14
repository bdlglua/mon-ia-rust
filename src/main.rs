use async_stream::stream;
use axum::{
    extract::{Multipart, Path as AxumPath, State},
    response::{
        sse::{Event, Sse},
        Html, Json,
    },
    routing::{delete, get, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use futures::{stream::Stream, StreamExt};
use genai::chat::{ChatMessage, ChatRequest, ChatStreamEvent};
use genai::Client;
use scraper::{Html as ScraperHtml, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const DEFAULT_MODEL: &str = "bdl:latest";
const EMBED_MODEL: &str = "nomic-embed-text";
const SESSIONS_DIR: &str = "sessions";
const DOCS_DIR: &str = "documents";
const MAX_RECENT_MESSAGES: usize = 16;

#[derive(Serialize, Deserialize, Clone)]
struct SavedMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct SessionData {
    id: String,
    title: String,
    summary: String,
    messages: Vec<SavedMessage>,
}

#[derive(Serialize)]
struct SessionMeta {
    id: String,
    title: String,
}

#[derive(Deserialize)]
struct ChatInput {
    session_id: String,
    message: String,
    model: Option<String>,
    use_image: Option<bool>,
}

#[derive(Deserialize)]
struct CreateSessionInput {
    title: Option<String>,
}

#[derive(Clone)]
struct DocChunk {
    filename: String,
    text: String,
    embedding: Vec<f32>,
}

struct AppState {
    client: Client,
    system_prompt: String,
    rag_index: Mutex<Vec<DocChunk>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::create_dir_all(SESSIONS_DIR);
    let _ = fs::create_dir_all(DOCS_DIR);

    let system_prompt = r#"Tu es BDL, un assistant IA personnel expert en Rust, en analyse documentaire et en raisonnement analytique.
    Avant de répondre ou d'utiliser un outil, réfléchis brièvement à voix haute entre les balises <thought> et </thought> pour planifier ta réponse.

    Tu as accès à 5 outils. Pour utiliser un outil, après ta réflexion </thought>, réponds UNIQUEMENT avec un objet JSON strict (sans texte autour, sans balises markdown) :
    1. Rechercher dans les documents locaux (RAG sémantique) : {"tool": "search_docs", "arg": "question ou concept recherché"}
    2. Lire un fichier précis : {"tool": "read_file", "arg": "chemin/fichier"}
    3. Écrire un fichier : {"tool": "write_file", "arg": "nom_fichier.txt", "content": "texte à écrire"}
    4. Rechercher sur le web : {"tool": "web_search", "arg": "mots clés de recherche"}
    5. Lire une page web : {"tool": "read_url", "arg": "https://exemple.com"}

    Si tu n'as pas besoin d'outil, écris ta réflexion entre <thought>...</thought> puis ta réponse finale claire en français (format Markdown)."#
    .to_string();

    println!("[RAG] Indexation sémantique des documents en cours...");
    let rag_index = build_rag_index(DOCS_DIR).await;
    println!("[RAG] {} segments vectorisés prêts.", rag_index.len());

    let state = Arc::new(AppState {
        client: Client::default(),
                         system_prompt,
                         rag_index: Mutex::new(rag_index),
    });

    let app = Router::new()
    .route("/", get(serve_ui))
    .route("/docs/{filename}", get(serve_doc))
    .route("/api/models", get(list_ollama_models))
    .route("/api/sessions", get(list_sessions).post(create_session))
    .route("/api/sessions/{id}", get(get_session).delete(delete_session))
    .route("/api/docs", get(list_docs))
    .route("/api/docs/{filename}", delete(delete_doc))
    .route("/api/upload", post(upload_file))
    .route("/api/reload-docs", post(reload_docs))
    .route("/api/chat", post(handle_chat_stream))
    .with_state(state);

    let addr = "127.0.0.1:3000";
    println!("Agent IA Rust complet prêt sur : http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn list_ollama_models() -> Json<Vec<String>> {
    Json(vec![DEFAULT_MODEL.to_string()])
}

/// Gestion des sessions multiples
async fn list_sessions() -> Json<Vec<SessionMeta>> {
    let mut list = Vec::new();
    if let Ok(entries) = fs::read_dir(SESSIONS_DIR) {
        for entry in entries.flatten() {
            if let Ok(data) = fs::read_to_string(entry.path()) {
                if let Ok(sess) = serde_json::from_str::<SessionData>(&data) {
                    list.push(SessionMeta {
                        id: sess.id,
                        title: sess.title,
                    });
                }
            }
        }
    }
    list.sort_by(|a, b| b.id.cmp(&a.id));
    Json(list)
}

async fn create_session(Json(payload): Json<CreateSessionInput>) -> Json<SessionData> {
    let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_millis();
    let id = format!("sess_{}", now);
    let title = payload
    .title
    .filter(|t| !t.trim().is_empty())
    .unwrap_or_else(|| "Nouvelle discussion".to_string());

    let sess = SessionData {
        id: id.clone(),
        title,
        summary: String::new(),
        messages: Vec::new(),
    };
    save_session(&sess);
    Json(sess)
}

async fn get_session(AxumPath(id): AxumPath<String>) -> Result<Json<SessionData>, axum::http::StatusCode> {
    load_session(&id)
    .map(Json)
    .ok_or(axum::http::StatusCode::NOT_FOUND)
}

async fn delete_session(AxumPath(id): AxumPath<String>) -> Json<&'static str> {
    if let Some(safe) = Path::new(&id).file_name().and_then(|n| n.to_str()) {
        let _ = fs::remove_file(Path::new(SESSIONS_DIR).join(format!("{}.json", safe)));
    }
    Json("OK")
}

/// Flux de conversation avec Mémoire Résumée, Vision conditionnelle et Cerveau
async fn handle_chat_stream(
    State(state): State<Arc<AppState>>,
                            Json(payload): Json<ChatInput>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let selected_model = payload
    .model
    .filter(|m| !m.trim().is_empty())
    .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let use_image = payload.use_image.unwrap_or(false);

    let s = stream! {
        let mut session = load_session(&payload.session_id).unwrap_or_else(|| SessionData {
            id: payload.session_id.clone(),
                                                                           title: "Discussion".to_string(),
                                                                           summary: String::new(),
                                                                           messages: Vec::new(),
        });

        // Met à jour automatiquement le titre de la session avec les premiers mots du 1er message
        if session.messages.is_empty() {
            let short_title: String = payload.message.chars().take(32).collect();
            session.title = if payload.message.chars().count() > 32 {
                format!("{}...", short_title)
            } else {
                short_title
            };
        }

        session.messages.push(SavedMessage {
            role: "user".to_string(),
                              content: payload.message.clone(),
        });

        // 1. Résumé automatique de l'ancien historique si la session dépasse MAX_RECENT_MESSAGES
        if session.messages.len() > MAX_RECENT_MESSAGES {
            yield Ok(Event::default().data(json!({ "type": "status", "text": "Compression de la mémoire à long terme..." }).to_string()));
            let overflow = session.messages.len() - MAX_RECENT_MESSAGES;
            let old_msgs: Vec<SavedMessage> = session.messages.drain(0..overflow).collect();
            let old_text = old_msgs
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

            if !old_text.is_empty() {
                let sum_prompt = format!(
                    "Résume de façon très concise en français les faits clés de cet historique en les fusionnant avec le résumé précédent.\nRésumé précédent : {}\nNouveaux échanges à intégrer :\n{}",
                    session.summary, old_text
                );
                let req = ChatRequest::new(vec![ChatMessage::user(sum_prompt)]);
                if let Ok(res) = state.client.exec_chat(&selected_model, req, None).await {
                    if let Some(new_sum) = res.content_text_as_str() {
                        session.summary = new_sum.to_string();
                    }
                }
            }
        }
        save_session(&session);

        // 2. Routage Vision conditionnel (uniquement si l'utilisateur a coché "Inclure l'image")
        if use_image {
            if let Some(img_b64) = get_uploaded_image_base64() {
                yield Ok(Event::default().data(json!({ "type": "status", "text": "Analyse visuelle par LLaVA..." }).to_string()));
                let vision_payload = json!({
                    "model": "llava:7b",
                    "messages": [{
                        "role": "user",
                        "content": format!("Décris précisément les éléments utiles pour répondre à : {}", payload.message),
                                           "images": [img_b64]
                    }],
                    "stream": false
                });

                let local_client = reqwest::Client::builder()
                .no_proxy()
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

                if let Ok(resp) = local_client
                    .post("http://localhost:11434/api/chat")
                    .json(&vision_payload)
                    .send()
                    .await
                    {
                        if let Ok(j) = resp.json::<Value>().await {
                            if let Some(desc) = j["message"]["content"].as_str() {
                                if let Some(last) = session.messages.last_mut() {
                                    last.content = format!(
                                        "[Observation visuelle (LLaVA) : {}]\n\nQuestion : {}",
                                                           desc, payload.message
                                    );
                                }
                                save_session(&session);
                            }
                        }
                    }
            }
        }

        loop {
            let chat_messages = build_context_messages(&state.system_prompt, &session);
            let chat_req = ChatRequest::new(chat_messages);
            let mut full_response = String::new();

            match state.client.exec_chat_stream(&selected_model, chat_req, None).await {
                Ok(mut chat_stream) => {
                    while let Some(event) = chat_stream.stream.next().await {
                        if let Ok(ChatStreamEvent::Chunk(chunk)) = event {
                            full_response.push_str(&chunk.content);
                            let data = json!({ "type": "chunk", "text": chunk.content }).to_string();
                            yield Ok(Event::default().data(data));
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Erreur Ollama ({}) : {}", selected_model, e);
                    yield Ok(Event::default().data(json!({ "type": "chunk", "text": err_msg }).to_string()));
                    break;
                }
            }

            session.messages.push(SavedMessage {
                role: "assistant".to_string(),
                                  content: full_response.clone(),
            });
            save_session(&session);

            let action_part = if let Some(idx) = full_response.find("</thought>") {
                full_response[idx + 10..].trim()
            } else {
                full_response.trim()
            };

            if action_part.starts_with('{') && action_part.ends_with('}') {
                if let Ok(parsed) = serde_json::from_str::<Value>(action_part) {
                    if let Some(tool) = parsed["tool"].as_str() {
                        let arg = parsed["arg"].as_str().unwrap_or("");
                        let content = parsed["content"].as_str().unwrap_or("");

                        let tool_msg = json!({ "type": "tool", "name": format!("{} ({})", tool, arg) }).to_string();
                        yield Ok(Event::default().data(tool_msg));

                        let tool_output = execute_tool(&state, tool, arg, content).await;
                        let feedback = format!("Résultat de l'outil '{}' :\n{}", tool, tool_output);

                        session.messages.push(SavedMessage {
                            role: "user".to_string(),
                                              content: feedback,
                        });
                        save_session(&session);
                        continue;
                    }
                }
            }

            yield Ok(Event::default().data(json!({ "type": "done", "title": session.title }).to_string()));
            break;
        }
    };

    Sse::new(s)
}

async fn execute_tool(state: &AppState, tool: &str, arg: &str, content: &str) -> String {
    match tool {
        "search_docs" => {
            let idx = state.rag_index.lock().unwrap().clone();
            search_rag_semantic(&idx, arg).await
        }
        "read_file" => fs::read_to_string(arg)
        .unwrap_or_else(|e| format!("Erreur de lecture : {}", e)),
        "write_file" => {
            let safe_name = Path::new(arg)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("sortie_agent.txt");
            match fs::write(safe_name, content) {
                Ok(_) => format!("Fichier '{}' écrit avec succès.", safe_name),
                Err(e) => format!("Erreur d'écriture : {}", e),
            }
        }
        "web_search" => search_duckduckgo(arg).await,
        "read_url" => scrape_webpage(arg).await,
        _ => "Outil inconnu.".to_string(),
    }
}

async fn get_embedding(text: &str) -> Vec<f32> {
    let client = reqwest::Client::builder()
    .no_proxy()
    .build()
    .unwrap_or_else(|_| reqwest::Client::new());

    let payload = json!({
        "model": EMBED_MODEL,
        "prompt": text
    });

    if let Ok(resp) = client
        .post("http://localhost:11434/api/embeddings")
        .json(&payload)
        .send()
        .await
        {
            if let Ok(j) = resp.json::<Value>().await {
                if let Some(arr) = j["embedding"].as_array() {
                    return arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                }
            }
        }
        Vec::new()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

async fn build_rag_index(dir_path: &str) -> Vec<DocChunk> {
    let mut chunks = Vec::new();
    let path = Path::new(dir_path);
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let file_path = entry.path();
            if !file_path.is_file() {
                continue;
            }
            let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("inconnu")
            .to_string();

            let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

            if ["jpg", "jpeg", "png", "webp", "gif"].contains(&ext.as_str()) {
                continue;
            }

            let extracted = if ext == "pdf" {
                pdf_extract::extract_text(&file_path).ok()
            } else {
                fs::read_to_string(&file_path).ok()
            };

            if let Some(content) = extracted {
                let mut current = String::new();
                for line in content.lines() {
                    let clean = line.trim();
                    if clean.is_empty() {
                        continue;
                    }
                    current.push_str(clean);
                    current.push(' ');
                    if current.len() >= 600 {
                        let text_chunk = current.trim().to_string();
                        let emb = get_embedding(&text_chunk).await;
                        chunks.push(DocChunk {
                            filename: filename.clone(),
                                    text: text_chunk,
                                    embedding: emb,
                        });
                        current.clear();
                    }
                }
                if !current.trim().is_empty() {
                    let text_chunk = current.trim().to_string();
                    let emb = get_embedding(&text_chunk).await;
                    chunks.push(DocChunk {
                        filename,
                        text: text_chunk,
                        embedding: emb,
                    });
                }
            }
        }
    }
    chunks
}

async fn search_rag_semantic(index: &[DocChunk], query: &str) -> String {
    if index.is_empty() {
        return "Aucun document textuel ou PDF indexé.".to_string();
    }

    let q_emb = get_embedding(query).await;
    let mut scored: Vec<(f32, &DocChunk)> = index
    .iter()
    .map(|chunk| {
        let sim = if !q_emb.is_empty() && !chunk.embedding.is_empty() {
            cosine_similarity(&q_emb, &chunk.embedding)
        } else {
            let q_low = query.to_lowercase();
            if chunk.text.to_lowercase().contains(&q_low) {
                0.5
            } else {
                0.0
            }
        };
        (sim, chunk)
    })
    .filter(|(sim, _)| *sim > 0.25)
    .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    if scored.is_empty() {
        return "Aucun passage sémantiquement proche trouvé dans vos documents.".to_string();
    }

    scored
    .into_iter()
    .take(4)
    .map(|(score, chunk)| format!("[Source : {} (Pertinence: {:.0}%)]\n{}", chunk.filename, score * 100.0, chunk.text))
    .collect::<Vec<String>>()
    .join("\n\n---\n\n")
}

async fn reload_docs(State(state): State<Arc<AppState>>) -> Json<usize> {
    let new_index = build_rag_index(DOCS_DIR).await;
    let count = new_index.len();
    *state.rag_index.lock().unwrap() = new_index;
    Json(count)
}

async fn upload_file(
    State(state): State<Arc<AppState>>,
                     mut multipart: Multipart,
) -> Json<Value> {
    let mut uploaded_names = Vec::new();
    let mut session_id = String::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_string();
        if field_name == "session_id" {
            if let Ok(txt) = field.text().await {
                session_id = txt;
            }
            continue;
        }

        if let Some(raw_name) = field.file_name() {
            let safe_name = Path::new(raw_name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("fichier.txt")
            .to_string();

            if let Ok(bytes) = field.bytes().await {
                let dest = Path::new(DOCS_DIR).join(&safe_name);
                if fs::write(&dest, &bytes).is_ok() {
                    uploaded_names.push(safe_name);
                }
            }
        }
    }

    let new_index = build_rag_index(DOCS_DIR).await;
    let total_chunks = new_index.len();
    *state.rag_index.lock().unwrap() = new_index;

    if !uploaded_names.is_empty() && !session_id.is_empty() {
        if let Some(mut sess) = load_session(&session_id) {
            sess.messages.push(SavedMessage {
                role: "upload".to_string(),
                               content: json!({
                                   "files": uploaded_names,
                                   "chunks": total_chunks
                               })
                               .to_string(),
            });
            save_session(&sess);
        }
    }

    Json(json!({
        "files": uploaded_names,
        "chunks": total_chunks
    }))
}

async fn list_docs() -> Json<Vec<String>> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(DOCS_DIR) {
        for entry in entries.flatten() {
            if entry.path().is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if !name.starts_with('.') {
                        files.push(name.to_string());
                    }
                }
            }
        }
    }
    files.sort();
    Json(files)
}

async fn delete_doc(
    State(state): State<Arc<AppState>>,
                    AxumPath(filename): AxumPath<String>,
) -> Json<usize> {
    if let Some(safe_name) = Path::new(&filename).file_name().and_then(|n| n.to_str()) {
        let _ = fs::remove_file(Path::new(DOCS_DIR).join(safe_name));
    }
    let new_index = build_rag_index(DOCS_DIR).await;
    let count = new_index.len();
    *state.rag_index.lock().unwrap() = new_index;
    Json(count)
}

async fn serve_doc(AxumPath(filename): AxumPath<String>) -> Result<Vec<u8>, axum::http::StatusCode> {
    let safe_name = Path::new(&filename)
    .file_name()
    .and_then(|n| n.to_str())
    .ok_or(axum::http::StatusCode::BAD_REQUEST)?;
    fs::read(Path::new(DOCS_DIR).join(safe_name)).map_err(|_| axum::http::StatusCode::NOT_FOUND)
}

async fn scrape_webpage(url: &str) -> String {
    let client = reqwest::Client::builder()
    .no_proxy()
    .user_agent("Mozilla/5.0 (compatible; RustAgent/1.0)")
    .build()
    .unwrap_or_else(|_| reqwest::Client::new());

    match client.get(url).send().await {
        Ok(resp) => match resp.text().await {
            Ok(html_content) => {
                let document = ScraperHtml::parse_document(&html_content);
                let selector = Selector::parse("p, h1, h2, h3, li").unwrap();
                let mut extracted = String::new();
                for element in document.select(&selector) {
                    let parts: Vec<&str> = element.text().collect();
                    let line = parts.join(" ").split_whitespace().collect::<Vec<&str>>().join(" ");
                    if line.len() > 20 {
                        extracted.push_str(&line);
                        extracted.push('\n');
                    }
                    if extracted.len() > 3500 {
                        break;
                    }
                }
                if extracted.is_empty() {
                    "Aucun texte lisible extrait.".to_string()
                } else {
                    extracted
                }
            }
            Err(e) => format!("Erreur HTML : {}", e),
        },
        Err(e) => format!("Erreur réseau : {}", e),
    }
}

async fn search_duckduckgo(query: &str) -> String {
    let client = reqwest::Client::builder()
    .no_proxy()
    .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
    .build()
    .unwrap_or_else(|_| reqwest::Client::new());

    let encoded = urlencoding::encode(query);
    let url = format!("https://duckduckgo.com{}", encoded);

    match client.get(&url).send().await {
        Ok(resp) => match resp.text().await {
            Ok(html_body) => {
                let document = ScraperHtml::parse_document(&html_body);
                let snippet_sel = Selector::parse(".result__snippet").unwrap();
                let title_sel = Selector::parse(".result__a").unwrap();

                let titles: Vec<String> = document
                .select(&title_sel)
                .map(|el| el.text().collect::<Vec<&str>>().join(" ").trim().to_string())
                .take(4)
                .collect();

                let snippets: Vec<String> = document
                .select(&snippet_sel)
                .map(|el| el.text().collect::<Vec<&str>>().join(" ").trim().to_string())
                .take(4)
                .collect();

                if snippets.is_empty() {
                    "Aucun résultat web trouvé.".to_string()
                } else {
                    titles
                    .into_iter()
                    .zip(snippets.into_iter())
                    .map(|(t, s)| format!("- **{}** : {}", t, s))
                    .collect::<Vec<String>>()
                    .join("\n")
                }
            }
            Err(e) => format!("Erreur lecture résultats : {}", e),
        },
        Err(e) => format!("Erreur réseau : {}", e),
    }
}

fn save_session(session: &SessionData) {
    if let Ok(json) = serde_json::to_string_pretty(session) {
        let path = Path::new(SESSIONS_DIR).join(format!("{}.json", session.id));
        let _ = fs::write(path, json);
    }
}

fn load_session(id: &str) -> Option<SessionData> {
    let safe = Path::new(id).file_name()?.to_str()?;
    let path = Path::new(SESSIONS_DIR).join(format!("{}.json", safe));
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn build_context_messages(system_prompt: &str, session: &SessionData) -> Vec<ChatMessage> {
    let mut full_sys = system_prompt.to_string();
    if !session.summary.is_empty() {
        full_sys.push_str("\n\n[Mémoire à long terme — Résumé des échanges précédents] :\n");
        full_sys.push_str(&session.summary);
    }

    let mut result = vec![ChatMessage::system(full_sys)];
    for m in &session.messages {
        if m.role == "upload" {
            continue;
        }
        match m.role.as_str() {
            "assistant" => result.push(ChatMessage::assistant(&m.content)),
            _ => result.push(ChatMessage::user(&m.content)),
        }
    }
    result
}

fn get_uploaded_image_base64() -> Option<String> {
    if let Ok(entries) = fs::read_dir(DOCS_DIR) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ["jpg", "jpeg", "png", "webp"].contains(&ext.to_lowercase().as_str()) {
                    if let Ok(bytes) = fs::read(&path) {
                        return Some(STANDARD.encode(&bytes));
                    }
                }
            }
        }
    }
    None
}

async fn serve_ui() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}

