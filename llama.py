import os
import time
import threading
import warnings

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ================================
# ENVIRONMENT
# ================================

warnings.filterwarnings("ignore")
load_dotenv()

# ================================
# PATHS
# ================================

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

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

Answer the question using ONLY the provided context.

Rules:

1. Do not use outside knowledge.
2. Use only information available in the context.
3. If the context does not contain the answer, reply:
"Answer not found in uploaded documents."
4. Do not add that sentence after already answering.
5. Give a complete explanation.
6. Use bullet points whenever useful.
7. Keep the answer focused and clear.


Context:

{context}


Question:

{question}


Answer:

"""


def set_custom_prompt():
    from langchain_core.prompts import PromptTemplate

    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
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


# ================================
# BUILD RAG CHAIN
# ================================

def build_rag_chain(llm, prompt, db):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    retriever = db.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        print("\n===== RETRIEVED DOCUMENTS =====")

        if not docs:
            return "No context found"

        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}")
            print(doc.page_content[:500])

        print("\n===============================")

        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ================================
# QA SYSTEM  (loads FAISS index)
# ================================

def qa_bot():
    """
    Loads the FAISS index from disk and builds the RAG chain.

    Returns (chain, error_message):
      - (chain, None) on success
      - (None, str)   if FAISS index is missing
    """
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(DB_FAISS_PATH):
        msg = (
            "FAISS vector database not found at '{}'. "
            "Please run create_vector_db.py locally, then commit the "
            "'vectorstore/db_faiss/' folder to your repository before "
            "deploying to Render.".format(DB_FAISS_PATH)
        )
        print("[ERROR] " + msg)
        return None, msg

    embeddings = get_embedder()

    print("Loading existing FAISS database...")

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("FAISS loaded successfully")

    llm = load_llm()
    prompt = set_custom_prompt()

    return build_rag_chain(llm, prompt, db), None


# ================================
# LAZY INITIALIZATION
# ================================
# QA_CHAIN is None until the first /ask request arrives.
# A threading.Lock prevents race conditions when Gunicorn uses
# multiple threads (even with --workers 1).

QA_CHAIN = None
_INIT_ERROR = None
_rag_lock = threading.Lock()


def get_rag_chain():
    """
    Returns (chain, error_message).

    The chain is built on the first call and cached for all subsequent
    requests so the heavy work only happens once per worker.
    """
    global QA_CHAIN, _INIT_ERROR

    # Fast path — already initialised (or already failed)
    if QA_CHAIN is not None or _INIT_ERROR is not None:
        return QA_CHAIN, _INIT_ERROR

    with _rag_lock:
        # Double-checked locking: re-check inside the lock
        if QA_CHAIN is not None or _INIT_ERROR is not None:
            return QA_CHAIN, _INIT_ERROR

        print("\nInitializing RAG System (lazy, first request)...")

        try:
            chain, error = qa_bot()
            if error:
                _INIT_ERROR = error
            else:
                QA_CHAIN = chain
                print("RAG System Ready\n")
        except Exception as exc:
            _INIT_ERROR = f"RAG system failed to initialise: {exc}"
            print(f"[ERROR] {_INIT_ERROR}")

    return QA_CHAIN, _INIT_ERROR


# ================================
# ROUTES
# ================================

@app.route("/")
def index():
    return render_template("open_ai_trail.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    query = data.get("query")

    if not query:
        return jsonify({"error": "Query required"}), 400

    chain, error = get_rag_chain()

    if error:
        return jsonify({"error": error}), 503

    start = time.time()

    answer = chain.invoke(query)

    end = time.time()

    response_time = round(end - start, 2)

    result = f"Response Time: {response_time} sec\n\n" + answer

    return jsonify(
        {
            "result": result.replace("\n", "<br>")
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    return jsonify(
        {
            "status": "success",
            "message": "Chat cleared",
        }
    )


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