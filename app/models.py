from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)


class SourceChunk(BaseModel):
    document: str
    page: int | None = None
    preview: str


class DocumentRecord(BaseModel):
    id: str
    name: str
    mime_type: str
    pages: int
    chunks: int
    original_size: int
    stored_size: int
    storage_path: str


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
    storage_mode: str
    authenticated: bool


class ConfigResponse(BaseModel):
    clerk_publishable_key: str
    auth_enabled: bool
    storage_mode: str
