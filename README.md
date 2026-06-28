# 📚 AI Document Chatbot using RAG (Retrieval-Augmented Generation)

A Retrieval-Augmented Generation (RAG) based AI chatbot that answers user questions from uploaded PDF documents using **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Groq LLM**, and **Flask**.

The chatbot retrieves the most relevant information from a collection of PDF documents and generates accurate answers using a Large Language Model (LLM).

---

# 🚀 Features

* 📄 Chat with multiple PDF documents
* 🔍 Semantic search using FAISS vector database
* 🧠 HuggingFace sentence embeddings
* ⚡ Fast inference powered by Groq LLM
* 💬 Interactive chatbot interface
* 📚 Retrieval-Augmented Generation (RAG)
* 🌐 Flask web application
* 🔄 Automatic document indexing
* 📝 Context-aware answers
* 🎯 Accurate document retrieval

---

# 🏗️ Project Architecture

```text
User
   │
   ▼
Flask Web Application
   │
   ▼
Receive User Question
   │
   ▼
FAISS Vector Database
   │
Retrieve Relevant Chunks
   │
   ▼
Prompt Template
   │
   ▼
Groq LLM
   │
   ▼
Answer Returned to User
```

---

# 🛠️ Technologies Used

* Python 3.10+
* Flask
* LangChain
* FAISS
* HuggingFace Embeddings
* Groq API
* Llama 3
* HTML
* CSS
* JavaScript

---

# 📂 Project Structure

```text
RAG/
│
├── data/
│   ├── AI Engineer Interview Prep Guide.pdf
│   ├── DATA SCIENCE INTERVIEW QUESTIONS.pdf
│   ├── Data science questions.pdf
│   ├── data-science-roadmap.pdf
│   ├── Finetuning LLM Dictionary.pdf
│   └── Top LLM Questions.pdf
│
├── templates/
│   └── open_ai_trail.html
│
├── llama.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Nagasai22222/rag-Agentic-chatbot.git

cd your-repository
```

---

## 2. Create a virtual environment

Windows

```bash
python -m venv myenv
```

Activate it

```bash
myenv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create a .env file

```text
GROQ_API_KEY=your_groq_api_key
```

---

## 5. Run the application

```bash
python llama.py
```

Open:

```text
http://127.0.0.1:8089
```

---

# 📖 How It Works

1. Load PDF documents.
2. Split documents into chunks.
3. Generate embeddings using HuggingFace.
4. Store embeddings in FAISS.
5. Retrieve the most relevant chunks.
6. Send retrieved context to Groq LLM.
7. Generate an answer based on the retrieved documents.
8. Display the response in the chatbot.

---

# 📸 Screenshots

## Chatbot Interface

> Add your chatbot screenshot here after uploading it to GitHub.

Example:

```text
screenshots/chatbot.png
```

---

# 🎯 Example Questions

* What is Fine Tuning?
* Explain LoRA.
* What is PEFT?
* What are Quantized Models?
* Explain Gradient Descent.
* What is Batch Size?
* What topics are covered in the Finetuning LLM Dictionary?

---

# 🔒 Environment Variables

Create a `.env` file in the project root.

```text
GROQ_API_KEY=your_api_key
```

**Do not upload your `.env` file to GitHub.**

---

# 📈 Future Improvements

* Upload PDFs from the web interface
* Chat history persistence
* User authentication
* Source citations in responses
* Streaming responses
* Docker support
* Cloud deployment
* Multi-user support

---

# 🤝 Contributing

Contributions are welcome.

Feel free to fork the repository and submit a pull request.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Mudunuri Naga Sai Srinivas**

B.Tech – Artificial Intelligence & Machine Learning

GitHub: https://github.com/Nagasai22222
