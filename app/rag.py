from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import DocumentRecord
from app.embeddings import get_embeddings


@dataclass
class RagStore:
    db: FAISS | None = None
    files: list[str] = field(default_factory=list)
    documents: list[DocumentRecord] = field(default_factory=list)
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
