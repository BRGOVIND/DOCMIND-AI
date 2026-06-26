import os
from pathlib import Path

from supabase import Client, create_client


def supabase_configured() -> bool:
    return bool(
        os.getenv("SUPABASE_URL")
        and os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        and os.getenv("SUPABASE_STORAGE_BUCKET")
    )


def storage_mode() -> str:
    return "supabase" if supabase_configured() else "local"


def _client() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    return create_client(url, key)


def store_file(path: Path, user_id: str, document_id: str) -> str:
    if not supabase_configured():
        return str(path)

    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")
    storage_path = f"{user_id}/{document_id}/{path.name}"

    with path.open("rb") as file:
        _client().storage.from_(bucket).upload(
            storage_path,
            file,
            file_options={"upsert": "true"},
        )

    return storage_path


def delete_user_prefix(user_id: str) -> None:
    if not supabase_configured():
        return

    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")
    storage = _client().storage.from_(bucket)
    items = storage.list(user_id)

    for item in items:
        name = item.get("name")
        if name:
            storage.remove([f"{user_id}/{name}"])


def delete_files(paths: list[str]) -> None:
    if not supabase_configured() or not paths:
        return

    bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")
    remote_paths = [path for path in paths if not Path(path).is_absolute()]
    if remote_paths:
        _client().storage.from_(bucket).remove(remote_paths)
