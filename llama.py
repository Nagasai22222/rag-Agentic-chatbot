import os
import time
import threading
import warnings
import logging
import hmac
from collections import deque

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import re

# --- Logging Setup ---
logger = logging.getLogger("rag_chatbot")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Configuration Constants & Environment Overrides ---
MAX_QUERY_LENGTH = int(os.environ.get("MAX_QUERY_LENGTH", 1000))
MAX_USER_ID_LENGTH = int(os.environ.get("MAX_USER_ID_LENGTH", 128))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
METRICS_AUTH_TOKEN = os.environ.get("METRICS_AUTH_TOKEN", None)
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", 60)) # seconds
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", 30))
MAX_TRACKED_RATE_LIMIT_CLIENTS = 1000

# --- Rate Limiter State ---
_rate_limit_lock = threading.Lock()
_rate_limit_store = {}

def is_rate_limited(client_id):
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(client_id)
        if timestamps is None:
            if len(_rate_limit_store) >= MAX_TRACKED_RATE_LIMIT_CLIENTS:
                expired_keys = [k for k, ts in _rate_limit_store.items() 
                                if not ts or (now - ts[-1]) > RATE_LIMIT_WINDOW]
                for k in expired_keys:
                    del _rate_limit_store[k]
                while len(_rate_limit_store) >= MAX_TRACKED_RATE_LIMIT_CLIENTS:
                    oldest_k = next(iter(_rate_limit_store))
                    del _rate_limit_store[oldest_k]

            timestamps = deque()
            _rate_limit_store[client_id] = timestamps

        while timestamps and (now - timestamps[0]) > RATE_LIMIT_WINDOW:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(RATE_LIMIT_WINDOW - (now - timestamps[0])) + 1
            return True, max(1, retry_after)

        timestamps.append(now)
        return False, 0

# --- Session Memory Configuration & Synchronization ---
MAX_HISTORY_TURNS = 6
MAX_SESSIONS = 100
GLOBAL_SESSIONS = {}
_session_lock = threading.Lock()

def get_session_history(user_id):
    with _session_lock:
        if user_id not in GLOBAL_SESSIONS:
            GLOBAL_SESSIONS[user_id] = []
        
        # LRU Cleanup
        if len(GLOBAL_SESSIONS) > MAX_SESSIONS and user_id not in GLOBAL_SESSIONS:
            oldest_key = next(iter(GLOBAL_SESSIONS))
            del GLOBAL_SESSIONS[oldest_key]
            
        return list(GLOBAL_SESSIONS[user_id])

def add_to_session(user_id, role, content):
    with _session_lock:
        if user_id not in GLOBAL_SESSIONS:
            GLOBAL_SESSIONS[user_id] = []
            
        if len(GLOBAL_SESSIONS) > MAX_SESSIONS and user_id not in GLOBAL_SESSIONS:
            oldest_key = next(iter(GLOBAL_SESSIONS))
            del GLOBAL_SESSIONS[oldest_key]
            
        history = GLOBAL_SESSIONS[user_id]
        history.append({"role": role, "content": content})
        if len(history) > MAX_HISTORY_TURNS * 2: # Keep 6 pairs of QA technically
            GLOBAL_SESSIONS[user_id] = history[-(MAX_HISTORY_TURNS * 2):]

DOMAIN_QUERY_MAP = {
    "rag": "retrieval augmented generation RAG architecture",
    "rag?": "retrieval augmented generation RAG architecture",
    "fine tuning": "fine-tuning model weights parameter adaptation",
    "fine-tuning": "fine-tuning model weights parameter adaptation",
    "the ai": "artificial intelligence LLM language models",
    "ai": "artificial intelligence LLM language models",
    "vector db": "vector databases embeddings similarity search FAISS",
    "vector database": "vector databases embeddings similarity search FAISS",
    "embeddings": "dense vector embeddings representations similarity",
    "tokenization": "tokenization tokens context window"
}

def extract_context_subject(history):
    if not history:
        return None
    
    last_user_msg = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), None)
    if not last_user_msg:
        return None
        
    text = last_user_msg.strip()
    prefixes = [
        "what is a ", "what is an ", "what is ", "what are ", 
        "explain ", "define ", "tell me about ", "describe ",
        "why is ", "how do ", "how does ", "compare ", "summarize "
    ]
    
    text_lower = text.lower().strip("?.! ")
    subject = text_lower
    for p in prefixes:
        if text_lower.startswith(p):
            subject = text_lower[len(p):].strip("?.! ")
            break
            
    for separator in [" and ", " or ", ",", ";"]:
        if separator in subject:
            subject = subject.split(separator)[0].strip()

    words = subject.split()
    if len(words) <= 5 and subject:
        return subject
        
    return " ".join(words[:5]) if words else None

def resolve_contextual_query(query, history):
    if not history or not query:
        return query, False

    subject = extract_context_subject(history)
    if not subject:
        return query, False

    query_lower = query.lower()

    # If the history subject is already explicitly present in the query, it is self-contained
    if subject.lower() in query_lower:
        return query, False

    pattern = r'\b(it|its|they|them|this|that|these|those)\b'
    pronoun_match = re.search(pattern, query_lower)

    is_followup_indicator = (
        pronoun_match is not None or 
        "previous" in query_lower or 
        "again" in query_lower or
        query_lower.startswith("why is") or
        query_lower.startswith("what are the") or
        query_lower.startswith("how is") or
        query_lower.startswith("how are")
    )

    if not is_followup_indicator:
        return query, False

    if pronoun_match:
        resolved = re.sub(pattern, subject, query, count=1, flags=re.IGNORECASE)
        if resolved.lower() != query_lower:
            return resolved, True

    if f"({subject})" not in query and subject not in query_lower:
        resolved = f"{query} ({subject})"
        return resolved, True

    return query, False

def expand_domain_query(query):
    if not query:
        return query, False

    clean_q = query.lower().strip("?.! ")
    if clean_q in DOMAIN_QUERY_MAP:
        expanded = DOMAIN_QUERY_MAP[clean_q]
        return f"{query} ({expanded})", True

    return query, False

def resolve_and_expand_query(query, history):
    start_t = time.perf_counter()
    res_query, is_contextual = resolve_contextual_query(query, history)
    final_query, is_expanded = expand_domain_query(res_query)
    res_time = time.perf_counter() - start_t
    return final_query, is_contextual, is_expanded, res_time

def resolve_conversational_query(query, history):
    final_q, _, _, _ = resolve_and_expand_query(query, history)
    return final_q

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
_cache_lock = threading.Lock()

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
    if not resolved_query:
        return None

    # Embed query outside the lock so we don't block other threads during model execution
    query_emb = np.array(embedder.embed_query(resolved_query))
    kb_ver = get_kb_version()
    now = time.time()
    
    with _cache_lock:
        keys_to_delete = [k for k, v in GLOBAL_SEMANTIC_CACHE.items() 
                          if now - v["timestamp"] > CACHE_TTL_SECONDS or v["kb_version"] != kb_ver]
        for k in keys_to_delete:
            del GLOBAL_SEMANTIC_CACHE[k]
            
        if not GLOBAL_SEMANTIC_CACHE:
            return None
            
        cache_snapshot = list(GLOBAL_SEMANTIC_CACHE.items())
        
        best_k = None
        best_sim = -1
        
        for k, v in cache_snapshot:
            sim = cosine_similarity(query_emb, v["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_k = k
                
        if best_sim >= CACHE_SIMILARITY_THRESHOLD and best_k is not None:
            if best_k in GLOBAL_SEMANTIC_CACHE:
                entry = GLOBAL_SEMANTIC_CACHE.pop(best_k)
                GLOBAL_SEMANTIC_CACHE[best_k] = entry
                payload = entry["payload"].copy() # avoid pointer mutating
                return payload
                
        return None

def save_to_cache(resolved_query, payload, embedder):
    if not payload.get("grounded") or "error" in payload:
        return
        
    query_emb = np.array(embedder.embed_query(resolved_query))
    kb_ver = get_kb_version()
    now = time.time()
    
    with _cache_lock:
        if len(GLOBAL_SEMANTIC_CACHE) >= MAX_CACHE_ENTRIES:
            oldest_k = next(iter(GLOBAL_SEMANTIC_CACHE))
            del GLOBAL_SEMANTIC_CACHE[oldest_k]
            record_metric("cache", "evictions", 1)
            
        GLOBAL_SEMANTIC_CACHE[resolved_query] = {
            "embedding": query_emb,
            "payload": payload,
            "timestamp": now,
            "kb_version": kb_ver
        }

# ================================
# PHASE 13 CITATION VERIFICATION ENGINE
# ================================

def verify_and_sanitize_citations(answer, sources_list, citation_sources_map=None):
    start_v = time.perf_counter()
    if not answer or not sources_list:
        return answer, {
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "sanitized_citations": 0,
            "verification_time": 0.0
        }
        
    pattern = r'\[Source\s+(\d+)(?:,\s*Pages?\s*([^\]]+))?\]'
    matches = list(re.finditer(pattern, answer, flags=re.IGNORECASE))
    
    total_citations = len(matches)
    valid_citations = 0
    invalid_citations = 0
    
    # 1. Chunk map: chunk_num (1..N) -> valid page string
    chunk_valid_pages = {}
    for item in sources_list:
        c_num = item.get("chunk")
        p_num = item.get("page")
        if c_num is not None:
            chunk_valid_pages[int(c_num)] = str(p_num).strip() if p_num is not None else "Unknown"
            
    # 2. Doc map: doc_id (1..M) -> list of valid page strings
    doc_valid_pages = {}
    if citation_sources_map:
        for s_file, c_data in citation_sources_map.items():
            d_id = c_data.get("id")
            pages = [str(p).strip() for p in c_data.get("pages", [])]
            if d_id is not None:
                doc_valid_pages[int(d_id)] = pages
                
    sanitized_answer = answer
    
    for match in reversed(matches):
        src_num = int(match.group(1))
        page_str = match.group(2)
        
        is_valid = False
        
        if src_num in chunk_valid_pages:
            valid_p = chunk_valid_pages[src_num]
            if page_str is None:
                is_valid = True
            else:
                p_str_clean = str(page_str).strip()
                if valid_p == "Unknown" or p_str_clean == valid_p:
                    is_valid = True
                elif citation_sources_map:
                    all_pages = []
                    for c_data in citation_sources_map.values():
                        all_pages.extend([str(p).strip() for p in c_data.get("pages", [])])
                    if p_str_clean in all_pages:
                        is_valid = True
                        
        if not is_valid and src_num in doc_valid_pages:
            valid_pages = doc_valid_pages[src_num]
            if page_str is None:
                is_valid = True
            else:
                p_str_clean = str(page_str).strip()
                if not valid_pages or "Unknown" in valid_pages or p_str_clean in valid_pages:
                    is_valid = True
                    
        if is_valid:
            valid_citations += 1
        else:
            invalid_citations += 1
            start_span, end_span = match.span()
            sanitized_answer = sanitized_answer[:start_span] + sanitized_answer[end_span:]
            
    if invalid_citations > 0:
        sanitized_answer = re.sub(r'\s+([.,;:!?])', r'\1', sanitized_answer)
        sanitized_answer = re.sub(r'[ \t]{2,}', ' ', sanitized_answer)
        
    v_time = time.perf_counter() - start_v
    
    record_metric("citations", "total_citations", total_citations)
    record_metric("citations", "valid_citations", valid_citations)
    record_metric("citations", "invalid_citations", invalid_citations)
    record_metric("citations", "sanitized_citations", invalid_citations)
    record_metric("citations", "verification_events", 1)
    try:
        with _metrics_lock:
            GLOBAL_METRICS["citations"]["total_verification_latency"] += v_time
    except Exception:
        pass
        
    stats = {
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "sanitized_citations": invalid_citations,
        "verification_time": round(v_time, 6)
    }
    
    return sanitized_answer, stats

# ================================
# FLASK APP
# ================================
# Flask app is created immediately so Gunicorn can bind to the port
# right away — before any heavy ML models are loaded.

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1 MB maximum payload limit
app.config['DEBUG'] = False
app.config['TESTING'] = False

if ALLOWED_ORIGINS == "*":
    CORS(app, resources={r"/*": {"origins": "*"}})
else:
    origins_list = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
    CORS(app, resources={r"/*": {"origins": origins_list}})

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "payload_too_large",
        "message": "Request payload exceeds maximum allowed size of 1 MB."
    }), 413

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({
        "error": "bad_request",
        "message": "The request body was malformed or missing required parameters."
    }), 400

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "server_error",
        "message": "An internal server error occurred."
    }), 500

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
            max_retries=1,
            api_key=os.environ.get("GROQ_API_KEY")
        )
        
        llm_fallback = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        qa_prompt = set_custom_prompt()

        return {
            "db": db, 
            "llm": llm, 
            "llm_fallback": llm_fallback,
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
    "query_resolution": {
        "total_resolutions": 0,
        "contextual_resolutions": 0,
        "domain_expansions": 0,
        "total_resolution_latency": 0.0
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
        "primary_generation_count": 0,
        "fallback_generation_count": 0,
        "fallback_success_count": 0,
        "fallback_failure_count": 0,
        "grounded_count": 0,
        "fast_trip_count": 0,
        "http_429_count": 0,
        "generation_failures": 0,
    },
    "citations": {
        "total_citations": 0,
        "valid_citations": 0,
        "invalid_citations": 0,
        "sanitized_citations": 0,
        "verification_events": 0,
        "total_verification_latency": 0.0
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
    if METRICS_AUTH_TOKEN:
        token = request.headers.get("X-Metrics-Token", "")
        if not token or not hmac.compare_digest(token, METRICS_AUTH_TOKEN):
            return jsonify({"error": "unauthorized", "message": "Invalid or missing metrics authentication token."}), 401
    try:
        with _metrics_lock:
            response = {
                "system": {
                    "uptime_seconds": round(time.time() - GLOBAL_METRICS["system"]["boot_time"], 2),
                    "total_requests": GLOBAL_METRICS["system"]["total_requests"],
                    "successful_requests": GLOBAL_METRICS["system"]["successful_requests"],
                    "errors": GLOBAL_METRICS["system"]["errors"]
                },
                "query_resolution": {
                    "total_resolutions": GLOBAL_METRICS["query_resolution"]["total_resolutions"],
                    "contextual_resolutions": GLOBAL_METRICS["query_resolution"]["contextual_resolutions"],
                    "domain_expansions": GLOBAL_METRICS["query_resolution"]["domain_expansions"]
                },
                "retrieval": GLOBAL_METRICS["retrieval"],
                "cache": GLOBAL_METRICS["cache"],
                "generation": GLOBAL_METRICS["generation"],
                "citations": {
                    "total_citations": GLOBAL_METRICS["citations"]["total_citations"],
                    "valid_citations": GLOBAL_METRICS["citations"]["valid_citations"],
                    "invalid_citations": GLOBAL_METRICS["citations"]["invalid_citations"],
                    "sanitized_citations": GLOBAL_METRICS["citations"]["sanitized_citations"],
                    "verification_events": GLOBAL_METRICS["citations"]["verification_events"]
                },
                "timing": {
                    "avg_query_resolution_latency": round(GLOBAL_METRICS["query_resolution"]["total_resolution_latency"] / max(1, GLOBAL_METRICS["query_resolution"]["total_resolutions"]), 6),
                    "avg_retrieval_latency": round(GLOBAL_METRICS["timing"]["total_retrieval_latency"] / max(1, GLOBAL_METRICS["timing"]["retrieval_events"]), 3),
                    "avg_generation_latency": round(GLOBAL_METRICS["timing"]["total_generation_latency"] / max(1, GLOBAL_METRICS["timing"]["generation_events"]), 3),
                    "avg_verification_latency": round(GLOBAL_METRICS["citations"]["total_verification_latency"] / max(1, GLOBAL_METRICS["citations"]["verification_events"]), 6),
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

@app.route("/ready", methods=["GET"])
def ready():
    components, error = get_rag_components()
    if components is not None and error is None:
        return jsonify({
            "status": "ready",
            "rag_initialized": True
        }), 200
    else:
        return jsonify({
            "status": "not_ready",
            "rag_initialized": False,
            "error": error or "System not initialized"
        }), 503

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)
    except Exception:
        data = None

    if not isinstance(data, dict):
        return jsonify({"error": "bad_request", "message": "Valid JSON request body required"}), 400

    raw_user_id = data.get("userId", "default_user")
    if not isinstance(raw_user_id, str):
        return jsonify({"error": "bad_request", "message": "userId must be a string"}), 400

    if len(raw_user_id) > MAX_USER_ID_LENGTH:
        return jsonify({
            "error": "bad_request",
            "message": f"userId exceeds maximum length of {MAX_USER_ID_LENGTH} characters."
        }), 400

    sanitized_user_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', raw_user_id)
    user_id = sanitized_user_id if sanitized_user_id else "default_user"

    raw_query = data.get("query")
    if not raw_query or not isinstance(raw_query, str):
        return jsonify({"error": "bad_request", "message": "Query parameter is required and must be a string"}), 400

    query = raw_query.strip()
    if not query:
        return jsonify({"error": "bad_request", "message": "Query parameter cannot be empty"}), 400

    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            "error": "bad_request",
            "message": f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters."
        }), 400

    client_ip = request.remote_addr or "unknown"
    client_id = f"{client_ip}:{user_id}"

    limited, retry_after = is_rate_limited(client_id)
    if limited:
        record_metric("generation", "http_429_count", 1)
        resp = jsonify({
            "error": "rate_limit_exceeded",
            "message": f"Too many requests. Please try again in {retry_after} seconds.",
            "retry_after": retry_after
        })
        resp.headers["Retry-After"] = str(retry_after)
        return resp, 429

    components, error = get_rag_components()
    if error:
        return jsonify({"error": error}), 503

    start_time = time.perf_counter()
    history = get_session_history(user_id)
    resolved_query, is_contextual, is_expanded, resolution_time = resolve_and_expand_query(query, history)
    history_used = (resolved_query != query)
    
    record_metric("query_resolution", "total_resolutions", 1)
    if is_contextual:
        record_metric("query_resolution", "contextual_resolutions", 1)
    if is_expanded:
        record_metric("query_resolution", "domain_expansions", 1)
    try:
        with _metrics_lock:
            GLOBAL_METRICS["query_resolution"]["total_resolution_latency"] += resolution_time
    except Exception:
        pass
    
    gen_history = history[-4:] if len(history) > 4 else history
    history_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in gen_history]) if gen_history else "No previous conversation."

    try:
        record_metric("system", "total_requests", 1)
        print(f"\n[QUERY] {resolved_query}")
        
        # PHASE 10 CACHE LOOKUP (Bypassed for contextual queries to prevent cross-session context contamination)
        cache_start = time.perf_counter()
        if not is_contextual:
            cached_payload = check_cache(resolved_query, components["embeddings"])
        else:
            cached_payload = None
            print("[CACHE] BYPASS (Contextual query relies on session history context)")
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
            print("[LLM] Generation started (Primary)")
            
            context_text = "\n\n".join(context_parts)
            chain_primary = (
                {"context": lambda x: context_text, "history": lambda x: history_text, "question": lambda x: resolved_query}
                | components["prompt"]
                | components["llm"]
            )
            
            try:
                record_metric("generation", "primary_generation_count", 1)
                res = chain_primary.invoke({})
                answer = res.content
            except Exception as e_primary:
                err_str = str(e_primary).lower()
                is_recoverable = '429' in err_str or 'rate limit' in err_str or 'too many requests' in err_str or 'rate_limit' in err_str or '503' in err_str or '500' in err_str or 'deadline' in err_str or 'connection' in err_str
                
                if not is_recoverable:
                    raise e_primary
                    
                print(f"[LLM] Primary model failed cleanly ({type(e_primary).__name__}), triggering fallback...")
                record_metric("generation", "fallback_generation_count", 1)
                
                chain_fallback = (
                    {"context": lambda x: context_text, "history": lambda x: history_text, "question": lambda x: resolved_query}
                    | components["prompt"]
                    | components["llm_fallback"]
                )
                
                try:
                    res = chain_fallback.invoke({})
                    answer = res.content
                    record_metric("generation", "fallback_success_count", 1)
                    print("[LLM] Fallback generated successfully")
                except Exception as e_fallback:
                    record_metric("generation", "fallback_failure_count", 1)
                    print(f"[LLM] Fallback model failed: {type(e_fallback).__name__}")
                    raise e_fallback
            
            generation_time = time.perf_counter() - generation_start
            print("[LLM] Generation sequence completed")
            
            record_metric("timing", "total_generation_latency", generation_time)
            record_metric("timing", "generation_events", 1)
            
            # Phase 1/3 Strict Override
            grounded = fallback_sentence not in answer
            
            if grounded:
                answer, citation_stats = verify_and_sanitize_citations(answer, sources_list, citation_sources_map)
                add_to_session(user_id, "user", query)
                add_to_session(user_id, "assistant", answer)
            else:
                citation_stats = {"total_citations": 0, "valid_citations": 0, "invalid_citations": 0, "sanitized_citations": 0, "verification_time": 0.0}
        else:
            record_metric("generation", "fast_trip_count", 1)
            generation_time = 0
            citation_stats = {"total_citations": 0, "valid_citations": 0, "invalid_citations": 0, "sanitized_citations": 0, "verification_time": 0.0}
            print("[LLM] Generation skipped (Fast-trip)")
            
        response_time = time.perf_counter() - start_time
        record_metric("timing", "total_request_latency", response_time)
        record_metric("system", "successful_requests", 1)
        
        final_payload = {
            "result": answer,
            "sources": sources_list,
            "citation_sources": citation_sources,
            "citation_stats": citation_stats,
            "retrieved_chunks": len(sources_list),
            "unique_documents": len(unique_docs_set),
            "response_time": round(response_time, 3),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "resolution_time": round(resolution_time, 6),
            "contextual_resolution": is_contextual,
            "domain_expansion": is_expanded,
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
        
        # P10: Inject Cache Write Boundary (Bypassed for contextual queries to prevent cross-session context contamination)
        if not is_contextual:
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
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}

    user_id = data.get("userId")
    if user_id and isinstance(user_id, str):
        if len(user_id) > MAX_USER_ID_LENGTH:
            return jsonify({"error": "bad_request", "message": f"userId exceeds maximum length of {MAX_USER_ID_LENGTH} characters."}), 400
        sanitized_user_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', user_id)
        if sanitized_user_id:
            with _session_lock:
                GLOBAL_SESSIONS.pop(sanitized_user_id, None)
            logger.info(f"[SESSION] Cleared memory for {sanitized_user_id}")
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