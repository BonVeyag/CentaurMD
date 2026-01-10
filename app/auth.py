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
from typing import Optional, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel


router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")
USER_ROOT_DIR = os.path.join(DATA_DIR, "users")
SIGNUP_ALLOWLIST_PATH = os.path.join(DATA_DIR, "signup_allowlist.json")

ADMIN_EMAIL = os.getenv("CENTAUR_ADMIN_EMAIL", "thapa.rajat@gmail.com").strip()
ADMIN_USERNAME = os.getenv("CENTAUR_ADMIN_USERNAME", "rajatthapa").strip().lower()
SIGNUP_MODE = os.getenv("CENTAUR_SIGNUP_MODE", "invite_only").strip().lower()
SMTP_CONFIG_PATH = os.getenv(
    "CENTAUR_SMTP_CONFIG_PATH",
    os.path.join(DATA_DIR, "smtp.json"),
).strip()
SMTP_REQUIRED = os.getenv("CENTAUR_SMTP_REQUIRED", "0").strip().lower() in {"1", "true", "yes"}

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
    is_admin: bool = False


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


def _is_admin_user(username: str, email: str) -> bool:
    uname = (username or "").strip().lower()
    eml = (email or "").strip().lower()
    if ADMIN_USERNAME and uname == ADMIN_USERNAME:
        return True
    if ADMIN_EMAIL and eml == ADMIN_EMAIL.strip().lower():
        return True
    return False


def _public_user(username: str, rec: Dict[str, Any]) -> AuthUser:
    email = rec.get("email", "")
    return AuthUser(
        username=username,
        email=email,
        created_at_utc=rec.get("created_at_utc", ""),
        is_admin=_is_admin_user(username, email),
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


def _env_first(*keys: str) -> str:
    for key in keys:
        val = os.getenv(key, "")
        if val is not None:
            val = val.strip()
            if val:
                return val
    return ""


def _format_from_header(name: str, email: str, fallback_email: str) -> str:
    addr = (email or "").strip() or (fallback_email or "").strip()
    if not addr:
        return ""
    display = (name or "").replace('"', "").strip()
    if display:
        return f"{display} <{addr}>"
    return addr


def _parse_from_header(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {"name": "", "email": ""}
    if "<" in raw and ">" in raw:
        name = raw.split("<", 1)[0].strip().strip('"')
        email = raw.split("<", 1)[1].split(">", 1)[0].strip()
        return {"name": name, "email": email}
    return {"name": "", "email": raw}


def _load_smtp_config() -> Dict[str, Any]:
    env_host = _env_first("CENTAUR_SMTP_HOST", "SMTP_HOST")
    env_user = _env_first("CENTAUR_SMTP_USER", "SMTP_USER")
    env_pass = _env_first("CENTAUR_SMTP_PASS", "SMTP_PASS")
    env_port = _env_first("CENTAUR_SMTP_PORT", "SMTP_PORT")
    env_tls = _env_first("CENTAUR_SMTP_TLS", "SMTP_TLS")
    env_ssl = _env_first("CENTAUR_SMTP_SSL", "SMTP_SSL")
    env_from = _env_first("CENTAUR_SMTP_FROM", "SMTP_FROM")
    env_from_name = _env_first("CENTAUR_SMTP_FROM_NAME", "SMTP_FROM_NAME")
    env_from_email = _env_first("CENTAUR_SMTP_FROM_EMAIL", "SMTP_FROM_EMAIL")
    env_admin = _env_first("CENTAUR_ADMIN_EMAIL", "ADMIN_EMAIL")
    env_timeout = _env_first("CENTAUR_SMTP_TIMEOUT", "SMTP_TIMEOUT")

    env_any = any([
        env_host, env_user, env_pass, env_port, env_tls, env_ssl, env_from, env_from_name, env_from_email, env_admin, env_timeout
    ])

    config: Dict[str, Any] = {
        "source": "env" if env_any else "none",
        "host": env_host,
        "port": int(env_port or 587),
        "user": env_user,
        "pass": env_pass,
        "tls": (env_tls or "true").strip().lower() in {"1", "true", "yes"},
        "ssl": (env_ssl or "false").strip().lower() in {"1", "true", "yes"},
        "from_name": env_from_name,
        "from_email": env_from_email,
        "from_raw": env_from,
        "admin_email": (env_admin or ADMIN_EMAIL).strip(),
        "timeout": int(env_timeout or 10),
    }

    if not env_any and SMTP_CONFIG_PATH and os.path.exists(SMTP_CONFIG_PATH):
        try:
            with open(SMTP_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config["source"] = "file"
                config["host"] = str(data.get("host", "")).strip() or config["host"]
                config["port"] = int(data.get("port", config["port"]))
                config["user"] = str(data.get("user", "")).strip() or config["user"]
                config["pass"] = str(data.get("pass", "")).strip() or config["pass"]
                if "tls" in data:
                    config["tls"] = bool(data.get("tls"))
                if "ssl" in data:
                    config["ssl"] = bool(data.get("ssl"))
                if "timeout" in data:
                    config["timeout"] = int(data.get("timeout"))
                config["from_raw"] = str(data.get("from", "")).strip() or config["from_raw"]
                config["from_name"] = str(data.get("from_name", "")).strip() or config["from_name"]
                config["from_email"] = str(data.get("from_email", "")).strip() or config["from_email"]
                config["admin_email"] = str(data.get("admin_email", "")).strip() or config["admin_email"]
        except Exception:
            pass

    if config["from_raw"]:
        parsed = _parse_from_header(config["from_raw"])
        config["from_name"] = config["from_name"] or parsed["name"]
        config["from_email"] = config["from_email"] or parsed["email"]

    config["from_header"] = _format_from_header(
        config.get("from_name", ""),
        config.get("from_email", ""),
        config.get("user", ""),
    )

    missing = []
    if not config.get("host"):
        missing.append("host")
    if not config.get("user"):
        missing.append("user")
    if not config.get("pass"):
        missing.append("pass")
    if not config.get("port"):
        missing.append("port")

    config["missing"] = missing
    config["configured"] = len(missing) == 0
    return config


def get_smtp_status() -> Dict[str, Any]:
    config = _load_smtp_config()
    return {
        "configured": bool(config.get("configured")),
        "source": config.get("source", "file"),
        "missing": config.get("missing", []),
        "tls": bool(config.get("tls", True)),
        "ssl": bool(config.get("ssl", False)),
        "from_set": bool(config.get("from_header")),
        "admin_email_configured": bool(config.get("admin_email")),
    }


def validate_smtp_config(required: Optional[bool] = None) -> Dict[str, Any]:
    config = _load_smtp_config()
    required_flag = SMTP_REQUIRED if required is None else bool(required)
    if required_flag and not config.get("configured"):
        raise RuntimeError(f"SMTP required but not configured. Missing: {config.get('missing', [])}")
    if not required_flag and not config.get("configured"):
        logger.warning(f"SMTP not configured. Missing: {config.get('missing', [])}")
    return config


def send_admin_email(subject: str, body: str, request: Request) -> Tuple[bool, str]:
    config = _load_smtp_config()
    host = config.get("host", "")
    admin_email = config.get("admin_email", "")
    if not host or not admin_email or not config.get("configured"):
        logger.warning("Admin email skipped: SMTP not configured.")
        return False, "SMTP_NOT_CONFIGURED"

    port = int(config.get("port", 587))
    smtp_user = config.get("user", "")
    smtp_pass = config.get("pass", "")
    use_tls = bool(config.get("tls", True))
    use_ssl = bool(config.get("ssl", False))
    from_addr = (config.get("from_header", "") or smtp_user).strip()
    timeout = int(config.get("timeout", 10))

    ip = request.client.host if request.client else ""
    user_agent = request.headers.get("User-Agent", "")
    now = _utc_now_iso()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = admin_email
    msg.set_content(
        "\n".join(
            [
                body.rstrip(),
                "",
                f"IP: {ip}",
                f"User-Agent: {user_agent}",
                f"Time (UTC): {now}",
            ]
        )
    )

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=timeout) as server:
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as server:
                if use_tls:
                    server.starttls(context=ssl.create_default_context())
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)
    except Exception as e:
        logger.warning(f"Admin email failed: {e.__class__.__name__}: {e}")
        return False, "SMTP_SEND_FAILED"
    return True, ""


def _send_signup_request_email(username: str, email: str, request: Request) -> None:
    body = "\n".join(
        [
            "A new signup request was received.",
            f"Username: {username}",
            f"Email: {email}",
        ]
    )
    send_admin_email("CentaurMD signup request", body, request)


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
def signup(payload: SignupPayload, request: Request):
    username = _normalize_username(payload.username)
    email = (payload.email or "").strip().lower()
    _validate_username(username)
    _validate_email(email)
    _validate_password(payload.password)

    if SIGNUP_MODE != "open":
        allowlist = _load_signup_allowlist()
        if username not in allowlist:
            _send_signup_request_email(username, email, request)
            raise HTTPException(
                status_code=403,
                detail="Signups are invite-only. Your request has been sent for approval.",
            )

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
            "approved": True,
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

    if SIGNUP_MODE != "open":
        allowlist = _load_signup_allowlist()
        if allowlist and username not in allowlist:
            raise HTTPException(status_code=403, detail="Account not approved.")

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
