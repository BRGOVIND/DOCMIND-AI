import os
from functools import lru_cache

import jwt
import requests
from fastapi import Header, HTTPException
from jwt import PyJWKClient


DEMO_USER_ID = "demo-user"


def clerk_publishable_key() -> str:
    return os.getenv("CLERK_PUBLISHABLE_KEY", "")


def clerk_configured() -> bool:
    return bool(os.getenv("CLERK_SECRET_KEY") and os.getenv("CLERK_JWT_ISSUER"))


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    issuer = os.getenv("CLERK_JWT_ISSUER", "").rstrip("/")
    if not issuer:
        raise ValueError("CLERK_JWT_ISSUER is missing.")
    return PyJWKClient(f"{issuer}/.well-known/jwks.json")


def get_current_user_id(authorization: str | None = Header(default=None)) -> str:
    if not clerk_configured():
        return DEMO_USER_ID

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in before using DocMind.")

    token = authorization.removeprefix("Bearer ").strip()
    issuer = os.getenv("CLERK_JWT_ISSUER", "").rstrip("/")

    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},
        )
    except (jwt.PyJWTError, requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session.") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid session payload.")

    return user_id
