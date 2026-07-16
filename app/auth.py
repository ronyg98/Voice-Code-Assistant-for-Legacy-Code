"""Authentication & role-aware access control.

OAuth2 password flow -> signed JWT (free, no external identity provider
needed; swap for Microsoft Entra ID later by replacing `authenticate` and
trusting Entra-issued tokens instead).

RBAC model (see app/config.py ROLES):
  - capabilities gate endpoints (index_repo, metrics, view_code, ...)
  - path scopes gate *content*: retrieval hits and file reads outside a
    user's allowed globs are filtered out BEFORE reaching the LLM or UI.
"""
import fnmatch
import time

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger

from app import db
from app.config import JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET, ROLES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

DEFAULT_USERS = [  # created on first boot so the demo works out of the box
    ("admin", "admin123", "admin"),
    ("dev", "dev123", "developer"),
    ("viewer", "viewer123", "viewer"),
]


def ensure_default_users() -> None:
    if db.count_users() == 0:
        for username, password, role in DEFAULT_USERS:
            db.create_user(username, hash_password(password), role)
        logger.warning("created default users (admin/dev/viewer) - change passwords "
                       "with scripts/create_user.py before real use")


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def authenticate(username: str, password: str) -> dict | None:
    user = db.get_user(username)
    if user and bcrypt.checkpw(password.encode(), user["pw_hash"]):
        return user
    return None


def create_token(username: str, role: str) -> str:
    payload = {"sub": username, "role": role,
               "exp": int(time.time()) + JWT_EXPIRE_MINUTES * 60}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail=f"invalid token: {exc}") from exc


async def current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    role = payload.get("role", "viewer")
    if role not in ROLES:
        role = "viewer"
    return {"username": payload["sub"], "role": role, **ROLES[role]}


def require(capability: str):
    """Dependency factory: 403 unless the user's role has the capability."""
    async def _check(user: dict = Depends(current_user)) -> dict:
        if capability not in user["capabilities"]:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                detail=f"role '{user['role']}' lacks '{capability}'")
        return user
    return _check


def path_allowed(user: dict, rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(rel_path, pat) or pat == "*"
               for pat in user.get("paths", []))


def filter_hits(user: dict, hits: list[dict]) -> tuple[list[dict], int]:
    """Drop retrieval hits the user may not see. Returns (kept, n_dropped)."""
    kept = [h for h in hits if path_allowed(user, h.get("path", ""))]
    return kept, len(hits) - len(kept)
