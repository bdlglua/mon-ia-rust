use base64::{engine::general_purpose::STANDARD, Engine as _};
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

use futures::{stream::Stream, StreamExt};
use genai::chat::{ChatMessage, ChatRequest, ChatStreamEvent};
use genai::Client;
use scraper::{Html as ScraperHtml, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    convert::Infallible,
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

const DEFAULT_MODEL: &str = "llama3.2:3b";
const HISTORY_FILE: &str = "historique.json";
const DOCS_DIR: &str = "documents";
const MAX_CONTEXT_MESSAGES: usize = 20;

#[derive(Serialize, Deserialize, Clone)]
struct SavedMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatInput {
    message: String,
    model: Option<String>,
}

#[derive(Clone)]
struct DocChunk {
    filename: String,
    text: String,
}

struct AppState {
    client: Client,
    history: Mutex<Vec<SavedMessage>>,
    system_prompt: String,
    rag_index: Mutex<Vec<DocChunk>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Indexer le dossier `documents/` pour le RAG local
    let rag_index = build_rag_index(DOCS_DIR);
    println!("[RAG] {} segments de documents indexés depuis '{}/'.", rag_index.len(), DOCS_DIR);

    let system_prompt = r#"Tu es un assistant agent en Rust. Tu as accès à 5 outils.
    Pour utiliser un outil, réponds UNIQUEMENT avec un objet JSON strict (sans texte autour, sans balises markdown) :
    1. Rechercher dans les documents locaux (RAG) : {"tool": "search_docs", "arg": "mots clés de la recherche"}
    2. Lire un fichier précis : {"tool": "read_file", "arg": "chemin/fichier"}
    3. Écrire un fichier : {"tool": "write_file", "arg": "nom_fichier.txt", "content": "texte à écrire"}
    4. Rechercher sur le web : {"tool": "web_search", "arg": "mots clés de recherche"}
    5. Lire une page web : {"tool": "read_url", "arg": "https://exemple.com"}

    Si l'utilisateur pose une question sur ses documents ou fichiers locaux, utilise TOUJOURS "search_docs" en premier.
    Sinon, réponds normalement en français en utilisant le format Markdown."#
    .to_string();

    let initial_history = load_history(&system_prompt);

    let state = Arc::new(AppState {
        client: Client::default(),
                         history: Mutex::new(initial_history),
                         system_prompt,
                         rag_index: Mutex::new(rag_index),
    });

    let app = Router::new()
    .route("/", get(serve_ui))
    .route("/marked.min.js", get(|| async {
        ([("content-type", "application/javascript")], include_str!("../marked.min.js"))
    }))
    .route("/docs/{filename}", get(serve_doc))
    .route("/api/models", get(list_ollama_models))
    .route("/api/docs", get(list_docs))
    .route("/api/docs/{filename}", delete(delete_doc))
    .route("/api/upload", post(upload_file))
    .route("/api/reload-docs", post(reload_docs))
    .route("/api/chat", post(handle_chat_stream))
    .route("/api/history", get(get_history))
    .route("/api/clear", post(clear_history))
    .with_state(state);


    let addr = "127.0.0.1:3000";
    println!("Agent IA Rust prêt sur : http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Récupère la liste des modèles installés sur Ollama
async fn list_ollama_models() -> Json<Vec<String>> {
    let mut models = Vec::new();
    if let Ok(output) = std::process::Command::new("ollama").arg("list").output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines().skip(1) {
            if let Some(name) = line.split_whitespace().next() {
                if !name.is_empty() {
                    models.push(name.to_string());
                }
            }
        }
    }
    if models.is_empty() {
        models.push(DEFAULT_MODEL.to_string());
    }
    Json(models)
}

/// Diffuse les tokens en direct vers le navigateur via SSE
async fn handle_chat_stream(
    State(state): State<Arc<AppState>>,
                            Json(payload): Json<ChatInput>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let selected_model = payload
    .model
    .filter(|m| !m.trim().is_empty())
    .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let s = stream! {
        {
            let mut hist = state.history.lock().unwrap();
            hist.push(SavedMessage {
                role: "user".to_string(),
                      content: payload.message.clone(),
            });
            save_history(&hist);
        }

        // Si une image est présente dans documents/, on l'envoie au modèle Vision d'Ollama
        // Routage automatique : si une image est présente, on utilise LLaVA pour l'analyser
        if let Some(img_b64) = get_uploaded_image_base64() {
            // 1. Étape Vision : LLaVA décrit ou extrait les informations de l'image
            let vision_prompt = format!(
                "Analyse cette image pour répondre à la demande suivante : {}",
                payload.message
            );

            let vision_payload = json!({
                "model": "llava:7b",
                "messages": [{
                    "role": "user",
                    "content": vision_prompt,
                    "images": [img_b64]
                }],
                "stream": false
            });

            let local_client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

            let vision_description = match local_client
            .post("http://localhost:11434/api/chat")
            .json(&vision_payload)
            .send()
            .await
            {
                Ok(resp) => match resp.json::<Value>().await {
                    Ok(j) => j["message"]["content"]
                    .as_str()
                    .unwrap_or("Impossible d'analyser l'image.")
                    .to_string(),
                    Err(e) => format!("Erreur lecture Vision : {}", e),
                },
                Err(e) => format!("Erreur Ollama Vision : {}", e),
            };

            // 2. Étape Cerveau (Llama 3.2) : on injecte ce que LLaVA a vu dans le contexte de Llama
            // Ainsi, Llama peut répondre en français fluide ET utiliser ses outils (ex: sauvegarder dans un fichier) !
            let combined_context = format!(
                "[Observation visuelle de l'image par LLaVA : {}]\n\nDemande de l'utilisateur : {}",
                vision_description, payload.message
            );

            {
                let mut hist = state.history.lock().unwrap();
                // Remplace le dernier message utilisateur par la version enrichie avec l'observation visuelle
                if let Some(last) = hist.last_mut() {
                    last.content = combined_context;
                }
                save_history(&hist);
            }

            // On ne fait pas `return` ici : on laisse la boucle `loop` continuer avec Llama 3.2 !
        }

        loop {
            let messages = {
                let hist = state.history.lock().unwrap();
                to_windowed_chat_messages(&hist, MAX_CONTEXT_MESSAGES)
            };

            let chat_req = ChatRequest::new(messages);
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
                    let data = json!({ "type": "chunk", "text": err_msg }).to_string();
                    yield Ok(Event::default().data(data));
                    break;
                }
            }

            {
                let mut hist = state.history.lock().unwrap();
                hist.push(SavedMessage {
                    role: "assistant".to_string(),
                          content: full_response.clone(),
                });
                save_history(&hist);
            }

            let trimmed = full_response.trim();
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
                    if let Some(tool) = parsed["tool"].as_str() {
                        let arg = parsed["arg"].as_str().unwrap_or("");
                        let content = parsed["content"].as_str().unwrap_or("");

                        let tool_msg = json!({ "type": "tool", "name": format!("{} ({})", tool, arg) }).to_string();
                        yield Ok(Event::default().data(tool_msg));

                        let tool_output = execute_tool(&state, tool, arg, content).await;
                        let feedback = format!("Résultat de l'outil '{}' :\n{}", tool, tool_output);

                        {
                            let mut hist = state.history.lock().unwrap();
                            hist.push(SavedMessage {
                                role: "user".to_string(),
                                      content: feedback,
                            });
                            save_history(&hist);
                        }
                        continue;
                    }
                }
            }

            yield Ok(Event::default().data(json!({ "type": "done" }).to_string()));
            break;
        }
    };

    Sse::new(s)
}

async fn get_history(State(state): State<Arc<AppState>>) -> Json<Vec<SavedMessage>> {
    let hist = state.history.lock().unwrap();
    Json(hist.clone())
}

async fn clear_history(State(state): State<Arc<AppState>>) -> Json<&'static str> {
    let mut hist = state.history.lock().unwrap();
    *hist = vec![SavedMessage {
        role: "system".to_string(),
        content: state.system_prompt.clone(),
    }];
    save_history(&hist);
    Json("OK")
}

async fn reload_docs(State(state): State<Arc<AppState>>) -> Json<usize> {
    let new_index = build_rag_index(DOCS_DIR);
    let count = new_index.len();
    let mut idx = state.rag_index.lock().unwrap();
    *idx = new_index;
    println!("[RAG] Index rechargé à chaud : {} segments.", count);
    Json(count)
}

async fn upload_file(
    State(state): State<Arc<AppState>>,
                     mut multipart: Multipart,
) -> Json<Value> {
    let mut uploaded_names = Vec::new();
    let _ = fs::create_dir_all(DOCS_DIR);

    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some(raw_name) = field.file_name() {
            let safe_name = Path::new(raw_name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("fichier_importe.txt")
            .to_string();

            if let Ok(bytes) = field.bytes().await {
                let dest = Path::new(DOCS_DIR).join(&safe_name);
                if fs::write(&dest, &bytes).is_ok() {
                    uploaded_names.push(safe_name);
                }
            }
        }
    }

    // Réindexation automatique immédiate après l'envoi
    let new_index = build_rag_index(DOCS_DIR);
    let total_chunks = new_index.len();
    *state.rag_index.lock().unwrap() = new_index;

    // 👇 Sauvegarde l'envoi dans l'historique pour qu'il persiste après un refresh (F5)
    if !uploaded_names.is_empty() {
        let mut hist = state.history.lock().unwrap();
        hist.push(SavedMessage {
            role: "upload".to_string(),
                  content: json!({
                      "files": uploaded_names,
                      "chunks": total_chunks
                  })
                  .to_string(),
        });
        save_history(&hist);
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
                    files.push(name.to_string());
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
    // Sécurité : empêche toute remontée de dossier (../)
    if let Some(safe_name) = Path::new(&filename).file_name().and_then(|n| n.to_str()) {
        let target = Path::new(DOCS_DIR).join(safe_name);
        let _ = fs::remove_file(target);
    }

    // Réindexation immédiate après suppression
    let new_index = build_rag_index(DOCS_DIR);
    let count = new_index.len();
    *state.rag_index.lock().unwrap() = new_index;
    Json(count)
}

async fn execute_tool(state: &AppState, tool: &str, arg: &str, content: &str) -> String {
    println!("[Agent Web] Exécution outil : {} -> {}", tool, arg);
    match tool {
        "search_docs" => {
            let idx = state.rag_index.lock().unwrap().clone();
            search_rag(&idx, arg)
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

/// Indexe tous les fichiers texte et PDF du dossier `documents/` par blocs de ~600 caractères
fn build_rag_index(dir_path: &str) -> Vec<DocChunk> {
    let mut chunks = Vec::new();
    let path = Path::new(dir_path);
    if !path.exists() {
        let _ = fs::create_dir_all(path);
        return chunks;
    }

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.is_file() {
                let filename = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("inconnu")
                .to_string();

                let is_pdf = file_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("pdf"))
                .unwrap_or(false);

                // Extraction PDF ou lecture texte classique
                let extracted_content = if is_pdf {
                    pdf_extract::extract_text(&file_path).ok()
                } else {
                    fs::read_to_string(&file_path).ok()
                };

                if let Some(content) = extracted_content {
                    let mut current_chunk = String::new();
                    for line in content.lines() {
                        let clean_line = line.trim();
                        if clean_line.is_empty() {
                            continue;
                        }
                        current_chunk.push_str(clean_line);
                        current_chunk.push(' ');
                        if current_chunk.len() >= 600 {
                            chunks.push(DocChunk {
                                filename: filename.clone(),
                                        text: current_chunk.trim().to_string(),
                            });
                            current_chunk.clear();
                        }
                    }
                    if !current_chunk.trim().is_empty() {
                        chunks.push(DocChunk {
                            filename,
                            text: current_chunk.trim().to_string(),
                        });
                    }
                }
            }
        }
    }
    chunks
}

/// Recherche les 4 segments les plus pertinents dans l'index RAG
fn search_rag(index: &[DocChunk], query: &str) -> String {
    if index.is_empty() {
        return "Le dossier 'documents/' est vide. Aucun document local indexé.".to_string();
    }

    let query_words: HashSet<String> = query
    .to_lowercase()
    .split(|c: char| !c.is_alphanumeric())
    .filter(|w| w.len() > 2)
    .map(|w| w.to_string())
    .collect();

    let mut scored: Vec<(usize, &DocChunk)> = index
    .iter()
    .map(|chunk| {
        let text_lower = chunk.text.to_lowercase();
        let score = query_words
        .iter()
        .filter(|w| text_lower.contains(w.as_str()))
        .count();
        (score, chunk)
    })
    .filter(|(score, _)| *score > 0)
    .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));

    if scored.is_empty() {
        return "Aucun passage correspondant trouvé dans vos documents locaux.".to_string();
    }

    scored
    .into_iter()
    .take(4)
    .map(|item| format!("[Source : {}]\n{}", item.1.filename, item.1.text))
    .collect::<Vec<String>>()
    .join("\n\n---\n\n")
}

async fn scrape_webpage(url: &str) -> String {
    let client = reqwest::Client::builder()
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
                    let text = parts.join(" ");
                    let clean: Vec<&str> = text.split_whitespace().collect();
                    let line = clean.join(" ");
                    if line.len() > 20 {
                        extracted.push_str(&line);
                        extracted.push('\n');
                    }
                    if extracted.len() > 3500 {
                        extracted.push_str("\n[... Contenu tronqué ...]");
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
    .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
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
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .take(4)
                .collect();

                let snippets: Vec<String> = document
                .select(&snippet_sel)
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .take(4)
                .collect();

                if snippets.is_empty() {
                    "Aucun résultat web trouvé pour cette recherche.".to_string()
                } else {
                    titles
                    .into_iter()
                    .zip(snippets.into_iter())
                    .map(|(t, s)| format!("- **{}** : {}", t, s))
                    .collect::<Vec<_>>()
                    .join("\n")
                }
            }
            Err(e) => format!("Erreur de lecture des résultats : {}", e),
        },
        Err(e) => format!("Erreur réseau (vérifiez votre connexion ou VPN) : {}", e),
    }
}

fn save_history(history: &[SavedMessage]) {
    if let Ok(json) = serde_json::to_string_pretty(history) {
        let _ = fs::write(HISTORY_FILE, json);
    }
}

fn load_history(system_prompt: &str) -> Vec<SavedMessage> {
    if Path::new(HISTORY_FILE).exists() {
        if let Ok(data) = fs::read_to_string(HISTORY_FILE) {
            if let Ok(loaded) = serde_json::from_str::<Vec<SavedMessage>>(&data) {
                if !loaded.is_empty() {
                    return loaded;
                }
            }
        }
    }
    vec![SavedMessage {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    }]
}

fn to_windowed_chat_messages(history: &[SavedMessage], max_messages: usize) -> Vec<ChatMessage> {
    let mut result = Vec::new();
    if history.is_empty() {
        return result;
    }
    if let Some(first) = history.first() {
        result.push(ChatMessage::system(&first.content));
    }
    let tail = &history[1..];
    let start = tail.len().saturating_sub(max_messages);
    for m in &tail[start..] {
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

async fn serve_doc(AxumPath(filename): AxumPath<String>) -> Result<Vec<u8>, axum::http::StatusCode> {
    let safe_name = Path::new(&filename)
    .file_name()
    .and_then(|n| n.to_str())
    .ok_or(axum::http::StatusCode::BAD_REQUEST)?;
    let path = Path::new(DOCS_DIR).join(safe_name);
    fs::read(path).map_err(|_| axum::http::StatusCode::NOT_FOUND)
}

async fn serve_ui() -> Html<&'static str> {
    Html(include_str!("../index.html"))
}
