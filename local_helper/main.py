import os
import sys
import time
import json
import secrets
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse

# Ensure repo root is on path so we can import app.transcription
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.transcription import transcribe_audio_bytes  # type: ignore

logger = logging.getLogger("centaur.local_helper")
logging.basicConfig(level=logging.INFO)

HELPER_VERSION = "0.1.0"
DEFAULT_PORT = int(os.getenv("LOCAL_HELPER_PORT", "57123"))

CONFIG_DIR = Path.home() / "Library" / "Application Support" / "CentaurHelper"
CONFIG_PATH = CONFIG_DIR / "config.json"

# In-memory pairing code cache
_PAIR_RECORD: Optional[Dict[str, Any]] = None


def ensure_config_dir():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def ensure_device_id(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not cfg.get("device_id"):
        cfg["device_id"] = secrets.token_urlsafe(12)
    return cfg


def load_config() -> Dict[str, Any]:
    ensure_config_dir()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    else:
        cfg = {}
    cfg.setdefault("paired", False)
    cfg.setdefault("helper_token", None)
    cfg.setdefault("backend_url", "https://centaurmd.ca")
    cfg.setdefault("paired_at", None)
    cfg = ensure_device_id(cfg)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    ensure_config_dir()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _detect_model_name() -> Optional[str]:
    return os.getenv("WHISPER_MODEL", "base")


app = FastAPI(title="Centaur Local Helper", version=HELPER_VERSION)


@app.get("/health")
async def health():
    cfg = load_config()
    return {
        "status": "ok",
        "version": HELPER_VERSION,
        "backend": "local_whisper",
        "model": _detect_model_name(),
        "paired": bool(cfg.get("paired")),
        "device_id": cfg.get("device_id"),
    }


def _require_auth(authorization: Optional[str]) -> None:
    cfg = load_config()
    if not cfg.get("paired") or not cfg.get("helper_token"):
        raise HTTPException(status_code=401, detail="unauthorized")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="unauthorized")
    token = authorization.split(None, 1)[1].strip()
    if token != cfg.get("helper_token"):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.post("/transcribe_chunk")
async def transcribe_chunk(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    _require_auth(authorization)
    if file is None:
        raise HTTPException(status_code=400, detail="missing file")
    try:
        payload = await file.read()
    except Exception as e:
        logger.exception("failed to read uploaded file")
        raise HTTPException(status_code=400, detail=f"failed to read file: {e}")

    if not payload:
        raise HTTPException(status_code=400, detail="empty file")

    start = time.time()
    try:
        result = transcribe_audio_bytes(payload, file.filename or "chunk.webm")
    except Exception as e:
        logger.exception("transcription error")
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")
    elapsed_ms = int((time.time() - start) * 1000)

    text = result.get("text") if isinstance(result, dict) else ""
    language = None
    if isinstance(result, dict):
        language = result.get("language") or result.get("language_used")
    return JSONResponse(
        {
            "text": text or "",
            "detected_language": language,
            "language_prob": result.get("language_prob") if isinstance(result, dict) else None,
            "ms": elapsed_ms,
        }
    )


def _gen_pair_code() -> str:
    # 6 digits with dash
    digits = f"{secrets.randbelow(1000000):06d}"
    return f"{digits[:3]}-{digits[3:]}"


@app.post("/pair/request")
async def pair_request():
    global _PAIR_RECORD
    cfg = load_config()
    pair_code = _gen_pair_code()
    _PAIR_RECORD = {
        "pair_code": pair_code,
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
        "device_id": cfg.get("device_id"),
    }
    return {
        "device_id": cfg.get("device_id"),
        "pair_code": pair_code,
        "expires_in_sec": 600,
    }


@app.post("/pair/confirm")
async def pair_confirm(body: Dict[str, Any]):
    global _PAIR_RECORD
    cfg = load_config()
    if not _PAIR_RECORD:
        raise HTTPException(status_code=400, detail="no pending pair request")
    pair_code = (body or {}).get("pair_code")
    helper_token = (body or {}).get("helper_token")
    if not pair_code or not helper_token:
        raise HTTPException(status_code=400, detail="pair_code and helper_token required")
    expires_at = _PAIR_RECORD.get("expires_at")
    if not expires_at or datetime.now(timezone.utc) > expires_at:
        _PAIR_RECORD = None
        raise HTTPException(status_code=400, detail="pair code expired")
    if pair_code != _PAIR_RECORD.get("pair_code"):
        raise HTTPException(status_code=400, detail="invalid pair code")
    cfg["paired"] = True
    cfg["helper_token"] = str(helper_token)
    cfg["paired_at"] = datetime.now(timezone.utc).isoformat()
    save_config(cfg)
    _PAIR_RECORD = None
    return {"status": "paired"}


@app.post("/pair/reset")
async def pair_reset():
    cfg = load_config()
    cfg["paired"] = False
    cfg["helper_token"] = None
    cfg["paired_at"] = None
    save_config(cfg)
    return {"status": "reset"}


def main():
    import uvicorn

    port = DEFAULT_PORT
    uvicorn.run(
        "local_helper.main:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
