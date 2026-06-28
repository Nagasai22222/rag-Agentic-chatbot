import os
import time
import warnings

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)


warnings.filterwarnings("ignore")
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")

# ================================
# PATHS
# ================================

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"



# ================================
# FLASK
# ================================

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

    return PromptTemplate(

        template=custom_prompt_template,

        input_variables=[
            "context",
            "question"
        ]

    )






# ================================
# LLM
# ================================

def load_llm():

    print("Loading Groq LLM...")

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=512,
    )








# ================================
# EMBEDDINGS
# ================================

def get_embedder():


    return HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )







# ================================
# CREATE VECTOR DATABASE
# ================================

def create_vector_db():


    print("\n========== BUILDING VECTOR DATABASE ==========")



    loader = DirectoryLoader(

        DATA_PATH,

        glob="*.pdf",

        loader_cls=PyPDFLoader

    )



    documents = loader.load()



    print(
        f"Loaded {len(documents)} pages from PDFs"
    )




    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=150

    )



    chunks = splitter.split_documents(documents)




    print(
        f"Created {len(chunks)} chunks"
    )




    embeddings = get_embedder()




    db = FAISS.from_documents(

        chunks,

        embeddings

    )



    os.makedirs(

        os.path.dirname(DB_FAISS_PATH),

        exist_ok=True

    )




    db.save_local(DB_FAISS_PATH)




    print(
        "FAISS database saved successfully"
    )

    print(
        "=============================================\n"
    )








# ================================
# BUILD RAG
# ================================

def build_rag_chain(llm, prompt, db):



    retriever = db.as_retriever(

        search_kwargs={
            "k":5
        }

    )




    def format_docs(docs):


        print(
            "\n===== RETRIEVED DOCUMENTS ====="
        )



        if not docs:

            return "No context found"




        for i,doc in enumerate(docs,1):


            print(
                f"\nDocument {i}"
            )


            print(
                doc.page_content[:500]
            )




        print(
            "\n==============================="
        )




        return "\n\n".join(

            doc.page_content

            for doc in docs

        )







    rag_chain = (


        RunnableParallel(

            {

                "context":
                retriever | format_docs,


                "question":
                RunnablePassthrough()

            }

        )


        |


        prompt


        |


        llm


        |


        StrOutputParser()


    )



    return rag_chain







# ================================
# QA SYSTEM
# ================================

def qa_bot():


    embeddings = get_embedder()




    if os.path.exists(DB_FAISS_PATH):



        print(
            "Loading existing FAISS database..."
        )



        db = FAISS.load_local(

            DB_FAISS_PATH,

            embeddings,

            allow_dangerous_deserialization=True

        )



        print(
            "FAISS loaded successfully"
        )




    else:



        create_vector_db()



        db = FAISS.load_local(

            DB_FAISS_PATH,

            embeddings,

            allow_dangerous_deserialization=True

        )







    llm = load_llm()



    prompt = set_custom_prompt()




    return build_rag_chain(

        llm,

        prompt,

        db

    )









# ================================
# INITIALIZE
# ================================

QA_CHAIN = None




def initialize_system():


    global QA_CHAIN



    print(
        "\nInitializing RAG System..."
    )



    QA_CHAIN = qa_bot()



    print(
        "RAG System Ready\n"
    )

# =========================================
# AUTO INITIALIZE FOR RENDER / GUNICORN
# =========================================

if QA_CHAIN is None:

    if not os.path.exists(DB_FAISS_PATH):

        print("Vector DB not found")

        create_vector_db()

    initialize_system()






# ================================
# ASK
# ================================

def final_result(query):


    return QA_CHAIN.invoke(query)







# ================================
# ROUTES
# ================================

@app.route("/")
def index():


    return render_template(
        "open_ai_trail.html"
    )







@app.route("/ask", methods=["POST"])
def ask():



    data=request.get_json()



    if not data:

        return jsonify(
            {
                "error":"No JSON received"
            }
        ),400




    query=data.get("query")



    if not query:


        return jsonify(
            {
                "error":"Query required"
            }
        ),400





    start=time.time()



    answer=final_result(query)



    end=time.time()




    response_time=round(

        end-start,

        2

    )




    result=(

        f"Response Time: {response_time} sec\n\n"

        +

        answer

    )




    return jsonify(

        {

            "result":

            result.replace(

                "\n",

                "<br>"

            )

        }

    )








@app.route("/reset",methods=["POST"])
def reset():


    return jsonify(

        {

            "status":"success",

            "message":"Chat cleared"

        }

    )









# ================================
# MAIN
# ================================

if __name__=="__main__":

    import os

    print(
        "Server starting..."
    )

    port = int(os.environ.get("PORT", 8089))

    print(
        f"Open on port: {port}"
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )