from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.auth import clerk_configured, clerk_publishable_key, get_current_user_id
from app.compressor import SUPPORTED_EXTENSIONS, compress_upload
from app.llm import ask_llm
from app.loader import load_document
from app.models import (
    AskRequest,
    AskResponse,
    ConfigResponse,
    DocumentRecord,
    StatusResponse,
    UploadResponse,
)
from app.rag import RagStore, create_vector_store, format_context, source_preview
from app.storage import delete_files, storage_mode, store_file

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
COMPRESSED_DIR = BASE_DIR / "data" / "compressed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DocMind AI",
    description="AI-powered document question-answering system using RAG.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
stores: dict[str, RagStore] = {}


def user_store(user_id: str) -> RagStore:
    if user_id not in stores:
        stores[user_id] = RagStore()
    return stores[user_id]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def config():
    return ConfigResponse(
        clerk_publishable_key=clerk_publishable_key(),
        auth_enabled=clerk_configured(),
        storage_mode=storage_mode(),
    )


@app.get("/status", response_model=StatusResponse)
def status(user_id: str = Depends(get_current_user_id)):
    store = user_store(user_id)
    return StatusResponse(
        ready=store.ready,
        files=store.files,
        pages=store.pages,
        chunks=store.chunks,
        storage_mode=storage_mode(),
        authenticated=clerk_configured(),
    )


@app.delete("/clear")
def clear_workspace(user_id: str = Depends(get_current_user_id)):
    store = user_store(user_id)
    delete_files([document.storage_path for document in store.documents])

    store.db = None
    store.files = []
    store.documents = []
    store.pages = 0
    store.chunks = 0

    for file_path in UPLOAD_DIR.glob(f"{user_id}_*"):
        file_path.unlink(missing_ok=True)

    for file_path in COMPRESSED_DIR.glob(f"{user_id}_*"):
        file_path.unlink(missing_ok=True)

    return {"message": "Workspace cleared."}


@app.post("/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id),
):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one document.")

    all_docs = []
    saved_files = []
    records = []

    for file in files:
        safe_name = Path(file.filename or "").name
        extension = Path(safe_name).suffix.lower()
        if not safe_name or extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, PNG, JPG, JPEG, and WEBP files are supported.",
            )

        document_id = uuid4().hex
        unique_name = f"{user_id}_{document_id}_{safe_name}"
        file_path = UPLOAD_DIR / unique_name
        raw_bytes = await file.read()
        file_path.write_bytes(raw_bytes)
        try:
            compressed_path = compress_upload(file_path, COMPRESSED_DIR)
            storage_path = store_file(compressed_path, user_id, document_id)
            docs = load_document(file_path)
        except ValueError as exc:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Upload processing failed: {exc}") from exc

        for doc in docs:
            doc.metadata["source"] = safe_name
            doc.metadata["document_id"] = document_id

        all_docs.extend(docs)
        saved_files.append(safe_name)
        records.append(
            DocumentRecord(
                id=document_id,
                name=safe_name,
                mime_type=file.content_type or "application/octet-stream",
                pages=len(docs),
                chunks=0,
                original_size=len(raw_bytes),
                stored_size=compressed_path.stat().st_size,
                storage_path=storage_path,
            )
        )

    db, chunk_count = create_vector_store(all_docs)
    chunk_count_per_file = max(1, chunk_count // max(1, len(records)))

    for record in records:
        record.chunks = chunk_count_per_file

    store = user_store(user_id)
    store.db = db
    store.files = saved_files
    store.documents = records
    store.pages = len(all_docs)
    store.chunks = chunk_count

    return UploadResponse(
        message="Documents uploaded and indexed.",
        files=saved_files,
        pages=len(all_docs),
        chunks=chunk_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, user_id: str = Depends(get_current_user_id)):
    store = user_store(user_id)
    if not store.ready or store.db is None:
        raise HTTPException(status_code=400, detail="Upload a document before asking questions.")

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


@app.get("/documents", response_model=list[DocumentRecord])
def documents(user_id: str = Depends(get_current_user_id)):
    return user_store(user_id).documents
