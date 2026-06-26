# DocMind AI

AI-powered PDF question-answering app using FastAPI, LangChain, Hugging Face embeddings, FAISS, OCR, and the Groq API.

## What You Built

Users can upload one or more PDF files, index them into a FAISS vector database, and ask questions like:

- Summarize this document.
- What are the key findings?
- Find all dates mentioned.
- Compare sections 2 and 5.
- Explain this in simple terms.

The app reads normal text-based PDFs directly. If a PDF page has little or no selectable text, it runs OCR with Tesseract so scanned PDFs and clear handwritten notes can still be indexed. Handwriting support is best-effort and depends on image quality.

The app can run in local demo mode without Clerk or Supabase keys. When those keys are added, requests are associated with authenticated Clerk users and compressed uploads can be stored in Supabase Storage.

## Folder Structure

```text
docmind-ai/
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- rag.py
|   |-- embeddings.py
|   |-- loader.py
|   |-- llm.py
|   |-- models.py
|   `-- templates/
|       `-- index.html
|-- data/
|   `-- .gitkeep
|-- .env
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- README.md
`-- requirements.txt
`-- requirements-cloud.txt
`-- requirements-ocr.txt
```

## Step 1: Create And Activate Virtual Environment

Open PowerShell inside the `docmind-ai` folder:

```powershell
cd C:\Users\LOQ\docmind-ai
python -m venv .venv
.venv\Scripts\activate
```

## Step 2: Install Requirements

```powershell
pip install -r requirements.txt
pip install -r requirements-cloud.txt
pip install -r requirements-ocr.txt
```

## Step 3: Add Your Groq API Key

Open `.env` and fill in your key:

```env
GROQ_API_KEY=your_real_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

You can change `GROQ_MODEL` later if your Groq console shows a different preferred model.

## Step 4: Run The App

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Step 5: Use The UI

1. Select one or more PDF files.
2. Click `Index documents`.
3. Ask a question in the right panel.
4. Read the answer and source snippets.

## File Snippets

### `app/loader.py`

Loads PDF pages using LangChain's `PyPDFLoader`.

```python
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str | Path):
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    file_name = Path(path).name
    for doc in docs:
        doc.metadata["source"] = file_name

    return docs
```

### `app/embeddings.py`

Creates Hugging Face sentence-transformer embeddings.

```python
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
```

### `app/rag.py`

Chunks documents, creates the FAISS vector store, and formats retrieved context.

```python
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.embeddings import get_embeddings


@dataclass
class RagStore:
    db: FAISS | None = None
    files: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0

    @property
    def ready(self) -> bool:
        return self.db is not None


def create_vector_store(docs: list[Document]) -> tuple[FAISS, int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, get_embeddings())
    return db, len(chunks)


def format_context(docs: list[Document]) -> str:
    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "unknown page"
        context_parts.append(
            f"[Source: {source}, {page_label}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def source_preview(doc: Document) -> dict:
    page = doc.metadata.get("page")
    return {
        "document": doc.metadata.get("source", "Unknown document"),
        "page": page + 1 if isinstance(page, int) else None,
        "preview": doc.page_content[:280].replace("\n", " ").strip(),
    }
```

### `app/llm.py`

Sends retrieved PDF context and the user's question to Groq.

```python
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def ask_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key)

    prompt = f"""
You are DocMind AI, a careful document question-answering assistant.
Use only the provided document context. If the context does not contain
enough information, say that clearly instead of guessing.

Context:
{context}

Question:
{question}

Answer in a clear, structured way. Mention document names or page numbers
when they help the user verify the answer.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using retrieved PDF context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
```

### `app/models.py`

Defines API request and response schemas.

```python
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)


class SourceChunk(BaseModel):
    document: str
    page: int | None = None
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    message: str
    files: list[str]
    pages: int
    chunks: int


class StatusResponse(BaseModel):
    ready: bool
    files: list[str]
    pages: int
    chunks: int
```

### `app/main.py`

Creates the FastAPI app, upload endpoint, ask endpoint, status endpoint, and UI route.

```python
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.llm import ask_llm
from app.loader import load_pdf
from app.models import AskRequest, AskResponse, StatusResponse, UploadResponse
from app.rag import RagStore, create_vector_store, format_context, source_preview

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DocMind AI",
    description="AI-powered document question-answering system using RAG.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
store = RagStore()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status():
    return StatusResponse(
        ready=store.ready,
        files=store.files,
        pages=store.pages,
        chunks=store.chunks,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")

    all_docs = []
    saved_files = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        safe_name = Path(file.filename).name
        unique_name = f"{uuid4().hex}_{safe_name}"
        file_path = UPLOAD_DIR / unique_name

        file_path.write_bytes(await file.read())
        docs = load_pdf(file_path)

        for doc in docs:
            doc.metadata["source"] = safe_name

        all_docs.extend(docs)
        saved_files.append(safe_name)

    db, chunk_count = create_vector_store(all_docs)

    store.db = db
    store.files = saved_files
    store.pages = len(all_docs)
    store.chunks = chunk_count

    return UploadResponse(
        message="Documents uploaded and indexed.",
        files=saved_files,
        pages=len(all_docs),
        chunks=chunk_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not store.ready or store.db is None:
        raise HTTPException(status_code=400, detail="Upload a PDF before asking questions.")

    docs = store.db.similarity_search(payload.question, k=4)
    context = format_context(docs)

    try:
        answer = ask_llm(context, payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {exc}") from exc

    return AskResponse(
        answer=answer,
        sources=[source_preview(doc) for doc in docs],
    )
```

### `.env`

Keep this file private.

```env
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

### `requirements.txt`

```text
fastapi
uvicorn[standard]
langchain
langchain-community
langchain-core
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
groq
python-multipart
jinja2
python-dotenv
```

### `requirements-cloud.txt`

```text
pyjwt[crypto]
requests
supabase
```

### `requirements-ocr.txt`

```text
pymupdf
pillow
pytesseract
```

## Optional Docker Run

Docker installs Tesseract OCR automatically, so scanned PDFs work inside the container.

```powershell
docker build -t docmind-ai .
docker run --env-file .env -p 8000:8000 docmind-ai
```

## API Endpoints

```text
GET  /              Browser UI
GET  /docs          FastAPI Swagger docs
GET  /health        Health check
GET  /status        Current indexed document status
POST /upload        Upload and index PDF files
POST /ask           Ask a question
```

Example `/ask` JSON body:

```json
{
  "question": "Summarize this document."
}
```
