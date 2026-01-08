from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import smtplib
import ssl
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from threading import Lock as ThreadLock
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel


router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")
USER_ROOT_DIR = os.path.join(DATA_DIR, "users")
SIGNUP_ALLOWLIST_PATH = os.path.join(DATA_DIR, "signup_allowlist.json")

ADMIN_EMAIL = os.getenv("CENTAUR_ADMIN_EMAIL", "thapa.rajat@gmail.com").strip()
SIGNUP_MODE = os.getenv("CENTAUR_SIGNUP_MODE", "invite_only").strip().lower()

logger = logging.getLogger(__name__)

USERS_LOCK = ThreadLock()
SESSIONS_LOCK = ThreadLock()
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")


class SignupPayload(BaseModel):
    username: str
    password: str
    email: str


class LoginPayload(BaseModel):
    username: str
    password: str


class AuthUser(BaseModel):
    username: str
    email: str
    created_at_utc: str


class AuthResponse(BaseModel):
    token: str
    user: AuthUser


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_username(username: str) -> str:
    return re.sub(r"\s+", "", (username or "").strip().lower())


def _validate_username(username: str) -> None:
    if not USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3-32 chars and use letters, numbers, dot, dash, underscore.",
        )


def _validate_email(email: str) -> None:
    e = (email or "").strip().lower()
    if "@" not in e or "." not in e.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")


def _validate_password(password: str) -> None:
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(USER_ROOT_DIR, exist_ok=True)


def _load_users() -> Dict[str, Any]:
    _ensure_dirs()
    if not os.path.exists(USERS_PATH):
        return {"version": 1, "users": {}}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"version": 1, "users": {}}
    if not isinstance(data, dict):
        return {"version": 1, "users": {}}
    data.setdefault("version", 1)
    if not isinstance(data.get("users"), dict):
        data["users"] = {}
    return data


def _save_users(data: Dict[str, Any]) -> None:
    _ensure_dirs()
    tmp_path = USERS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, USERS_PATH)


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return base64.b64encode(dk).decode("ascii")


def _verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
    except Exception:
        return False
    calc = _hash_password(password, salt)
    return hmac.compare_digest(calc, hash_b64 or "")


def _public_user(username: str, rec: Dict[str, Any]) -> AuthUser:
    return AuthUser(
        username=username,
        email=rec.get("email", ""),
        created_at_utc=rec.get("created_at_utc", ""),
    )


def _ensure_user_dir(username: str) -> None:
    path = os.path.join(USER_ROOT_DIR, username)
    os.makedirs(path, exist_ok=True)


def _load_signup_allowlist() -> set[str]:
    if not os.path.exists(SIGNUP_ALLOWLIST_PATH):
        return set()
    try:
        with open(SIGNUP_ALLOWLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    raw = data.get("usernames", [])
    if not isinstance(raw, list):
        return set()
    return {str(u).strip().lower() for u in raw if str(u).strip()}


def _send_signup_request_email(username: str, email: str, request: Request) -> None:
    host = os.getenv("CENTAUR_SMTP_HOST", "").strip()
    if not host or not ADMIN_EMAIL:
        logger.warning("Signup request email skipped: SMTP host or admin email not configured.")
        return

    port = int(os.getenv("CENTAUR_SMTP_PORT", "587"))
    smtp_user = os.getenv("CENTAUR_SMTP_USER", "").strip()
    smtp_pass = os.getenv("CENTAUR_SMTP_PASS", "").strip()
    use_tls = os.getenv("CENTAUR_SMTP_TLS", "true").strip().lower() in {"1", "true", "yes"}
    use_ssl = os.getenv("CENTAUR_SMTP_SSL", "false").strip().lower() in {"1", "true", "yes"}
    from_addr = os.getenv("CENTAUR_SMTP_FROM", ADMIN_EMAIL).strip() or ADMIN_EMAIL

    ip = request.client.host if request.client else ""
    user_agent = request.headers.get("User-Agent", "")
    now = _utc_now_iso()

    msg = EmailMessage()
    msg["Subject"] = "CentaurMD signup request"
    msg["From"] = from_addr
    msg["To"] = ADMIN_EMAIL
    msg.set_content(
        "\n".join(
            [
                "A new signup request was received.",
                f"Username: {username}",
                f"Email: {email}",
                f"IP: {ip}",
                f"User-Agent: {user_agent}",
                f"Time (UTC): {now}",
            ]
        )
    )

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context) as server:
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port) as server:
                if use_tls:
                    server.starttls(context=ssl.create_default_context())
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
    except Exception as e:
        logger.warning(f"Signup request email failed: {e}")


def user_storage_dir(username: str) -> str:
    _ensure_dirs()
    _ensure_user_dir(username)
    return os.path.join(USER_ROOT_DIR, username)


def user_billing_dir(username: str) -> str:
    base = user_storage_dir(username)
    path = os.path.join(base, "billing")
    os.makedirs(path, exist_ok=True)
    return path


def _create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with SESSIONS_LOCK:
        SESSIONS[token] = {
            "username": username,
            "created_at": now,
            "expires_at": now + SESSION_TTL_SECONDS,
        }
    return token


def _get_session(token: str) -> Optional[str]:
    if not token:
        return None
    now = int(time.time())
    with SESSIONS_LOCK:
        sess = SESSIONS.get(token)
        if not sess:
            return None
        if sess.get("expires_at", 0) < now:
            SESSIONS.pop(token, None)
            return None
        return sess.get("username")


def _extract_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return request.headers.get("X-Auth-Token", "").strip()


def require_user(request: Request) -> AuthUser:
    token = _extract_token(request)
    username = _get_session(token)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    with USERS_LOCK:
        data = _load_users()
        rec = data.get("users", {}).get(username)
    if not rec:
        raise HTTPException(status_code=401, detail="User not found.")
    return _public_user(username, rec)


@router.post("/auth/signup", response_model=AuthResponse)
def signup(payload: SignupPayload):
    username = _normalize_username(payload.username)
    email = (payload.email or "").strip().lower()
    _validate_username(username)
    _validate_email(email)
    _validate_password(payload.password)

    with USERS_LOCK:
        data = _load_users()
        users = data.setdefault("users", {})
        if username in users:
            raise HTTPException(status_code=409, detail="Username already exists.")
        for u, rec in users.items():
            if (rec.get("email") or "").strip().lower() == email:
                raise HTTPException(status_code=409, detail="Email already in use.")

        salt = secrets.token_bytes(16)
        users[username] = {
            "email": email,
            "password_hash": _hash_password(payload.password, salt),
            "salt": base64.b64encode(salt).decode("ascii"),
            "created_at_utc": _utc_now_iso(),
            "updated_at_utc": _utc_now_iso(),
        }
        _save_users(data)

    _ensure_user_dir(username)
    token = _create_session(username)
    return {"token": token, "user": _public_user(username, users[username])}


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginPayload):
    username = _normalize_username(payload.username)
    if not username or not payload.password:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    with USERS_LOCK:
        data = _load_users()
        rec = data.get("users", {}).get(username)

    if not rec:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not _verify_password(payload.password, rec.get("salt", ""), rec.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = _create_session(username)
    return {"token": token, "user": _public_user(username, rec)}


@router.get("/auth/me")
def me(user: AuthUser = Depends(require_user)):
    return {"user": user}


@router.post("/auth/logout")
def logout(request: Request):
    token = _extract_token(request)
    if token:
        with SESSIONS_LOCK:
            SESSIONS.pop(token, None)
    return {"ok": True}
