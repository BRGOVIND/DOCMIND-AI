# 🧠 DocMind AI

> AI-powered document intelligence — upload PDFs and images, ask questions, get cited answers.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

---

## ✨ Features

- 📄 **Multi-format upload** — PDF, PNG, JPG, WEBP
- 🔍 **OCR support** — scanned documents and handwritten notes via Tesseract
- 🧩 **RAG pipeline** — Retrieval-Augmented Generation with FAISS vector search
- 🤖 **Groq LLM** — fast, free-tier AI answers using `llama3-8b-8192`
- 🔐 **Clerk authentication** — secure per-user workspaces
- ☁️ **Supabase storage** — cloud file storage with per-user isolation
- 🐳 **Docker ready** — one command to run locally
- 🚀 **Railway deployed** — live at [d0cmindai.up.railway.app](https://d0cmindai.up.railway.app)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| LLM | Groq API (llama3-8b-8192) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Auth | Clerk |
| Storage | Supabase |
| OCR | Tesseract |
| Deployment | Docker + Railway |

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/BRGOVIND/DOCMIND-AI.git
cd DOCMIND-AI
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Fill in your keys in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192

CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key_here
CLERK_SECRET_KEY=your_clerk_secret_key_here
CLERK_JWT_ISSUER=https://your-clerk-domain.clerk.accounts.dev

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
SUPABASE_STORAGE_BUCKET=documents
```

### 3. Run with Docker

```bash
docker build -t docmind-ai .
docker run -d --name docmind-ai-app --env-file .env -p 8000:8000 docmind-ai
```

### 4. Open the app

```
http://localhost:8000
```

---

## 📁 Project Structure

```
docmind-ai/
├── app/
│   ├── auth.py          # Clerk JWT authentication
│   ├── compressor.py    # PDF and image compression
│   ├── embeddings.py    # Sentence transformer embeddings
│   ├── llm.py           # Groq LLM integration
│   ├── loader.py        # Document loading and OCR
│   ├── main.py          # FastAPI routes
│   ├── models.py        # Pydantic models
│   ├── rag.py           # RAG pipeline and FAISS store
│   ├── storage.py       # Supabase storage integration
│   └── templates/
│       └── index.html   # Frontend UI
├── Dockerfile
├── requirements.txt
├── requirements-ocr.txt
├── requirements-cloud.txt
└── .env.example
```

---

## 🔑 Getting API Keys

| Service | Link |
|---|---|
| Groq | [console.groq.com](https://console.groq.com) |
| Clerk | [dashboard.clerk.com](https://dashboard.clerk.com) |
| Supabase | [supabase.com](https://supabase.com) |

---

## 🌐 Live Demo

**[https://d0cmindai.up.railway.app](https://d0cmindai.up.railway.app)**

---

## 📝 How It Works

```
User uploads PDF/image
        ↓
OCR + text extraction (Tesseract / pypdf)
        ↓
Text split into chunks
        ↓
Chunks embedded (sentence-transformers)
        ↓
Stored in FAISS vector index
        ↓
User asks a question
        ↓
Top-k relevant chunks retrieved
        ↓
Groq LLM generates cited answer
        ↓
Answer displayed with source snippets
```

---

## 📄 License

MIT
