import os
import time
import threading
import warnings

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import re

# --- Session Memory Configuration ---
MAX_HISTORY_TURNS = 6
MAX_SESSIONS = 100
GLOBAL_SESSIONS = {}

def get_session_history(user_id):
    if user_id not in GLOBAL_SESSIONS:
        GLOBAL_SESSIONS[user_id] = []
    
    # LRU Cleanup
    if len(GLOBAL_SESSIONS) > MAX_SESSIONS:
        oldest_key = next(iter(GLOBAL_SESSIONS))
        del GLOBAL_SESSIONS[oldest_key]
        
    return GLOBAL_SESSIONS[user_id]

def add_to_session(user_id, role, content):
    history = get_session_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_TURNS * 2: # Keep 6 pairs of QA technically
        GLOBAL_SESSIONS[user_id] = history[-(MAX_HISTORY_TURNS * 2):]

def extract_subject(text):
    text = text.lower().strip("?.! ")
    prefixes = ["what is a ", "what is an ", "what is ", "what are ", "explain ", "define ", "tell me about ", "describe "]
    for p in prefixes:
        if text.startswith(p):
            return text[len(p):].strip()
    return text

def resolve_conversational_query(query, history):
    if not history:
        return query
        
    last_user_msg = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), None)
    if not last_user_msg:
        return query
        
    subject = extract_subject(last_user_msg)
    
    # Pronouns indicating follow up
    pattern = r'\b(it|its|they|them|this|that|these|those)\b'
    
    query_lower = query.lower()
    if re.search(pattern, query_lower) or "previous" in query_lower or "again" in query_lower:
        resolved = re.sub(pattern, subject, query, count=1, flags=re.IGNORECASE)
        if resolved.lower() == query_lower:
            resolved = f"{query} ({subject})"
        return resolved
        
    return query

# ================================
# ENVIRONMENT
# ================================

warnings.filterwarnings("ignore")
load_dotenv()

# --- Configuration & Credentials ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
RETRIEVAL_K = 15
MAX_CONTEXT_CHUNKS = 5
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "1.50"))

# ================================
# PATHS
# ================================

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

import numpy as np

# ================================
# PHASE 10 SEMANTIC CACHE
# ================================
MAX_CACHE_ENTRIES = 100
CACHE_TTL_SECONDS = 3600
CACHE_SIMILARITY_THRESHOLD = 0.95

GLOBAL_SEMANTIC_CACHE = {}

def get_kb_version():
    try:
        return os.path.getmtime(os.path.join(DB_FAISS_PATH, "index.faiss"))
    except:
        return 0

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return dot / (n1 * n2) if n1 and n2 else 0

def check_cache(resolved_query, embedder):
    kb_ver = get_kb_version()
    now = time.time()
    
    keys_to_delete = [k for k, v in GLOBAL_SEMANTIC_CACHE.items() 
                      if now - v["timestamp"] > CACHE_TTL_SECONDS or v["kb_version"] != kb_ver]
    for k in keys_to_delete:
        del GLOBAL_SEMANTIC_CACHE[k]
        
    if not GLOBAL_SEMANTIC_CACHE:
        return None
        
    query_emb = np.array(embedder.embed_query(resolved_query))
    best_k = None
    best_sim = -1
    
    for k, v in GLOBAL_SEMANTIC_CACHE.items():
        sim = cosine_similarity(query_emb, v["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_k = k
            
    if best_sim >= CACHE_SIMILARITY_THRESHOLD and best_k is not None:
        entry = GLOBAL_SEMANTIC_CACHE.pop(best_k)
        GLOBAL_SEMANTIC_CACHE[best_k] = entry
        payload = entry["payload"].copy() # avoid pointer mutating
        return payload
        
    return None

def save_to_cache(resolved_query, payload, embedder):
    if not payload.get("grounded") or "error" in payload:
        return
        
    if len(GLOBAL_SEMANTIC_CACHE) >= MAX_CACHE_ENTRIES:
        oldest_k = next(iter(GLOBAL_SEMANTIC_CACHE))
        del GLOBAL_SEMANTIC_CACHE[oldest_k]
        record_metric("cache", "evictions", 1)
        
    query_emb = np.array(embedder.embed_query(resolved_query))
    GLOBAL_SEMANTIC_CACHE[resolved_query] = {
        "embedding": query_emb,
        "payload": payload,
        "timestamp": time.time(),
        "kb_version": get_kb_version()
    }

# ================================
# FLASK APP
# ================================
# Flask app is created immediately so Gunicorn can bind to the port
# right away — before any heavy ML models are loaded.

app = Flask(__name__)
CORS(app)

# ================================
# FINAL RAG PROMPT
# ================================

custom_prompt_template = """

You are a document-based AI assistant.
You are an expert, truthful AI assistant strictly grounded in the provided documents.

1. You must answer the Question ONLY using the facts from the Retrieved Documents.
2. The Conversation History is provided ONLY to help you understand pronouns or follow-up references in the Question (e.g., "it", "they"). Do NOT use the Conversation History as a source of factual knowledge.
3. If the retrieved context contains a directly relevant fact, definition, explanation, or statement that can answer the user's question, answer using that information, even if the retrieved passage is short, fragmented, or presented as a bullet point or glossary entry.
4. When the context provides only partial information, answer only what can be supported by that information and clearly avoid unsupported details.
5. CRITICAL: Never refuse to answer if the context contains ANY fragments or operational details relevant to the topic. Only return the fallback sentence if the topic is completely absent or unsupported by all chunks. If it is unsupported, return exactly:
"I couldn't find enough information in the provided documents to answer this question reliably."
6. Give a concise but useful answer. For simple questions, prefer a direct answer. For procedural questions, use numbered steps. For questions involving several concepts, use short bullet points. Do not pad with generic explanations.
7. CRITICAL CITATION RULE: When incorporating a fact from a specific evidence block, append an inline citation EXACTLY like [Source X, Page Y]. Example: "Fine-tuning adjusts model weights for specific tasks [Source 1, Page 7]." Multiple sources look like: "... tasks [Source 1, Page 7] [Source 2, Page 8]."
8. Do NOT invent source numbers or pages. Use ONLY the Source, Document, and Page identifiers strictly provided in the logical blocks below.

Conversation History (Context only):
{history}

Context:

{context}

Question:

{question}
"""


def set_custom_prompt():
    from langchain_core.prompts import PromptTemplate

    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["history", "context", "question"]
    )


# ================================
# LLM  (imported lazily)
# ================================

def load_llm():
    from langchain_groq import ChatGroq

    print("Loading Groq LLM...")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
    )


# ================================
# EMBEDDINGS  (imported lazily)
# ================================

def get_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Loading HuggingFace embeddings...")

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def qa_bot():
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq

    try:
        if not os.path.exists(DB_FAISS_PATH):
            return None, "Vector database not found. Please ingest documents first."
            
        embeddings = get_embedder()
        print("Loading existing FAISS database...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS loaded successfully")
        
        print("Loading Groq LLM...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        qa_prompt = set_custom_prompt()

        return {
            "db": db, 
            "llm": llm, 
            "prompt": qa_prompt,
            "embeddings": embeddings
        }, None
    except Exception as e:
        return None, str(e)


# ================================
# EAGER INITIALIZATION
# ================================

QA_COMPONENTS = None
_INIT_ERROR = None
_rag_lock = threading.Lock()

def initialize_rag_system():
    global QA_COMPONENTS, _INIT_ERROR

    with _rag_lock:
        if QA_COMPONENTS is not None or _INIT_ERROR is not None:
            return

        print("\nStarting RAG system initialization...")
        init_start = time.perf_counter()

        try:
            components, error = qa_bot()
            if error:
                _INIT_ERROR = error
            else:
                QA_COMPONENTS = components
                elapsed = time.perf_counter() - init_start
                print(f"RAG initialization completed in {elapsed:.3f} seconds.")
                print("RAG system ready.\n")
        except Exception as exc:
            _INIT_ERROR = f"RAG system failed to initialise: {exc}"
            print(f"[ERROR] {_INIT_ERROR}")

# Perform Eager Initialization 
initialize_rag_system()

def get_rag_components():
    return QA_COMPONENTS, _INIT_ERROR


# ================================
# PHASE 11 OBSERVABILITY METRICS
# ================================
GLOBAL_METRICS = {
    "system": {
        "boot_time": time.time(),
        "total_requests": 0,
        "successful_requests": 0,
        "errors": 0,
    },
    "retrieval": {
        "total_chunks_retrieved": 0,
    },
    "cache": {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
    },
    "generation": {
        "total_generations": 0,
        "grounded_count": 0,
        "fast_trip_count": 0,
        "http_429_count": 0,
        "generation_failures": 0,
    },
    "timing": {
        "total_retrieval_latency": 0.0,
        "total_generation_latency": 0.0,
        "total_request_latency": 0.0,
        "retrieval_events": 0,
        "generation_events": 0
    }
}
_metrics_lock = threading.Lock()

def record_metric(category, key, increment=1):
    try:
        with _metrics_lock:
            GLOBAL_METRICS[category][key] += increment
    except Exception:
        pass

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        with _metrics_lock:
            response = {
                "system": {
                    "uptime_seconds": round(time.time() - GLOBAL_METRICS["system"]["boot_time"], 2),
                    "total_requests": GLOBAL_METRICS["system"]["total_requests"],
                    "successful_requests": GLOBAL_METRICS["system"]["successful_requests"],
                    "errors": GLOBAL_METRICS["system"]["errors"]
                },
                "retrieval": GLOBAL_METRICS["retrieval"],
                "cache": GLOBAL_METRICS["cache"],
                "generation": GLOBAL_METRICS["generation"],
                "timing": {
                    "avg_retrieval_latency": round(GLOBAL_METRICS["timing"]["total_retrieval_latency"] / max(1, GLOBAL_METRICS["timing"]["retrieval_events"]), 3),
                    "avg_generation_latency": round(GLOBAL_METRICS["timing"]["total_generation_latency"] / max(1, GLOBAL_METRICS["timing"]["generation_events"]), 3),
                    "avg_total_latency": round(GLOBAL_METRICS["timing"]["total_request_latency"] / max(1, GLOBAL_METRICS["system"]["total_requests"]), 3)
                }
            }
            return jsonify(response)
    except Exception:
        return jsonify({"error": "metrics failure"}), 500

# ================================
# ROUTES
# ================================

@app.route("/")
def index():
    return render_template("open_ai_trail.html")

@app.route("/health", methods=["GET"])
def health():
    if QA_COMPONENTS is not None:
        return jsonify({
            "status": "ok",
            "rag_ready": True
        }), 200
    else:
        return jsonify({
            "status": "error",
            "rag_ready": False,
            "details": _INIT_ERROR
        }), 503


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    user_id = data.get("userId", "default_user")
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query required"}), 400

    components, error = get_rag_components()
    if error:
        return jsonify({"error": error}), 503

    start_time = time.perf_counter()
    history = get_session_history(user_id)
    resolved_query = resolve_conversational_query(query, history)
    history_used = (resolved_query != query)
    
    gen_history = history[-4:] if len(history) > 4 else history
    history_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in gen_history]) if gen_history else "No previous conversation."

    try:
        record_metric("system", "total_requests", 1)
        print(f"\n[QUERY] {resolved_query}")
        
        # PHASE 10 CACHE LOOKUP
        cache_start = time.perf_counter()
        cached_payload = check_cache(resolved_query, components["embeddings"])
        cache_lookup_time = time.perf_counter() - cache_start
        
        if cached_payload is not None:
            record_metric("cache", "hits", 1)
            print("[CACHE] HIT! Bypassing Langchain/Groq architecture.")
            response_time = time.perf_counter() - start_time
            cached_payload["response_time"] = round(response_time, 3)
            cached_payload["cache_hit"] = True
            cached_payload["cache_lookup_time"] = round(cache_lookup_time, 3)
            
            add_to_session(user_id, "user", query)
            add_to_session(user_id, "assistant", cached_payload["result"])
            
            record_metric("timing", "total_request_latency", response_time)
            record_metric("system", "successful_requests", 1)
            return jsonify(cached_payload)
            
        record_metric("cache", "misses", 1)
        print("[CACHE] MISS.")
        
        # Retrieval Stage
        retrieval_start = time.perf_counter()
        docs_and_scores = components["db"].similarity_search_with_score(resolved_query, k=int(os.getenv("RETRIEVAL_K", "15")))
        retrieval_time = time.perf_counter() - retrieval_start
        
        record_metric("timing", "total_retrieval_latency", retrieval_time)
        record_metric("timing", "retrieval_events", 1)
        
        relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "1.50"))
        diversity_penalty = float(os.getenv("DIVERSITY_PENALTY", "0.15"))
        max_context_chunks = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
        
        pool = []
        for doc, score in docs_and_scores:
            if score <= relevance_threshold:
                pool.append({"doc": doc, "initial_score": score, "current_score": score})
        
        sources_list = []
        unique_docs_set = set()
        context_parts = []
        citation_sources_map = {}
        fallback_sentence = "I couldn't find enough information in the provided documents to answer this question reliably."
        
        while pool and len(sources_list) < max_context_chunks:
            pool.sort(key=lambda x: x["current_score"])
            best = pool.pop(0)
            doc, score = best["doc"], best["initial_score"]
            
            meta = doc.metadata or {}
            source_file = os.path.basename(str(meta.get("source", "Unknown source")))
            page_num = meta.get("page", "Unknown")
            chunk_num = len(sources_list) + 1
            
            sources_list.append({"source": source_file, "page": page_num, "chunk": chunk_num, "score": round(float(score), 4), "chunk_text": doc.page_content})
            unique_docs_set.add(source_file)
            
            if source_file not in citation_sources_map:
                citation_sources_map[source_file] = {"id": len(citation_sources_map) + 1, "source": source_file, "pages": [], "chunks": [], "scores": []}
            if page_num not in citation_sources_map[source_file]["pages"]:
                citation_sources_map[source_file]["pages"].append(page_num)
            citation_sources_map[source_file]["chunks"].append(chunk_num)
            citation_sources_map[source_file]["scores"].append(round(float(score), 4))
            
            context_parts.append(f"[Source {chunk_num}]\nDocument: {source_file}\nPage: {page_num}\nChunk: {chunk_num}\n\n{doc.page_content}")
            
            # Apply penalty to remaining candidates of the same document
            for item in pool:
                item_source = os.path.basename(str(item["doc"].metadata.get("source", "")))
                if item_source == source_file:
                    item["current_score"] += diversity_penalty
            
        citation_sources = list(citation_sources_map.values())
        record_metric("retrieval", "total_chunks_retrieved", len(sources_list))
        
        if not sources_list:
            answer = fallback_sentence
            grounded = False
            print("[GROUNDING] Grounded: False (No relevant chunks)")
        else:
            grounded = True
            print(f"[GROUNDING] Grounded: {grounded}")
        
        if grounded:
            record_metric("generation", "grounded_count", 1)
            record_metric("generation", "total_generations", 1)
            generation_start = time.perf_counter()
            print("[LLM] Generation started")
            
            formatted_prompt = components["prompt"].format(
                history=history_text,
                context="\n\n".join(context_parts),
                question=resolved_query
            )
            response = components["llm"].invoke(formatted_prompt)
            answer = response.content
            
            # Phase 1/3 Strict Override
            grounded = fallback_sentence not in answer
            
            generation_time = time.perf_counter() - generation_start
            print("[LLM] Generation completed")
            
            record_metric("timing", "total_generation_latency", generation_time)
            record_metric("timing", "generation_events", 1)
            
            if grounded:
                add_to_session(user_id, "user", query)
                add_to_session(user_id, "assistant", answer)
        else:
            record_metric("generation", "fast_trip_count", 1)
            generation_time = 0
            print("[LLM] Generation skipped (Fast-trip)")
            
        response_time = time.perf_counter() - start_time
        record_metric("timing", "total_request_latency", response_time)
        record_metric("system", "successful_requests", 1)
        
        final_payload = {
            "result": answer,
            "sources": sources_list,
            "citation_sources": citation_sources,
            "retrieved_chunks": len(sources_list),
            "unique_documents": len(unique_docs_set),
            "response_time": round(response_time, 3),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "grounded": grounded,
            "rag_initialized": True,
            "conversation_id": user_id,
            "resolved_query": resolved_query,
            "history_used": history_used,
            "history_char_count": len(history_text),
            "generation_history_turns": len(gen_history) // 2,
            "cache_hit": False,
            "cache_lookup_time": round(cache_lookup_time, 3)
        }
        
        # P10: Inject Cache Write Boundary
        save_to_cache(resolved_query, final_payload, components["embeddings"])
        
        return jsonify(final_payload)
    except Exception as e:
        record_metric("system", "errors", 1)
        err_str = str(e).lower()
        print(f"[ERROR] Ask processing failed: {e}")
        if '429' in err_str or 'rate limit' in err_str or 'too many requests' in err_str or 'rate_limit' in err_str:
            record_metric("generation", "http_429_count", 1)
            return jsonify({
                "error": "rate_limit",
                "message": "The language model service is temporarily rate limited. Please try again shortly.",
                "retryable": True
            }), 429
            
        record_metric("generation", "generation_failures", 1)
        return jsonify({
            "error": "provider_error",
            "message": "The language model service encountered an error. Please try again.",
            "retryable": False
        }), 500


@app.route("/reset", methods=["POST"])
def reset():
    data = request.json or {}
    user_id = data.get("userId")
    if user_id:
        GLOBAL_SESSIONS.pop(user_id, None)
        print(f"[SESSION] Cleared memory for {user_id}")
    return jsonify({"status": "success", "message": "Session reset initialized."})


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8089))

    print("Server starting...")
    print(f"Running on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )