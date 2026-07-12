"""
create_vector_db.py
====================
Offline script to build the FAISS vector database from the PDF files
in the `data/` directory.

Run this ONCE on your local machine before deploying to Render:

    python create_vector_db.py

After it completes, commit the generated index to Git:

    git add vectorstore/db_faiss/
    git commit -m "Add FAISS index for Render deployment"
    git push

DO NOT run this on Render — the PDF files are not present there and
the free-tier instance lacks the RAM to build the index.
"""

import os
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

# ================================
# PATHS
# ================================

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"


def create_vector_db():
    """Load PDFs, split into chunks, embed, and save FAISS index."""

    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    print("\n========== BUILDING VECTOR DATABASE ==========")

    # ---- Load PDFs ----
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"Data directory '{DATA_PATH}' not found. "
            "Place your PDF files in the 'data/' folder and try again."
        )

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No PDF documents found in '{DATA_PATH}'. "
            "Add PDF files and run this script again."
        )

    print(f"Loaded {len(documents)} pages from PDFs")

    # ---- Chunk ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # ---- Embed ----
    print("Loading HuggingFace embeddings (this may take a moment)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Build FAISS index ----
    print("Building FAISS index...")

    db = FAISS.from_documents(chunks, embeddings)

    # ---- Save ----
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

    db.save_local(DB_FAISS_PATH)

    print("FAISS database saved to:", DB_FAISS_PATH)
    print("=============================================\n")
    print("Next steps:")
    print("  git add vectorstore/db_faiss/")
    print('  git commit -m "Add FAISS index"')
    print("  git push")


if __name__ == "__main__":
    create_vector_db()
