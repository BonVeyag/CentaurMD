from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router as api_router
from app.auth import router as auth_router, require_user, validate_smtp_config, get_smtp_status
import logging
import os
import subprocess
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("centaurweb")

app = FastAPI(
    title="CentaurWeb",
    version="0.1.0"
)

# CORS – safe for now; can tighten once auth is added
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.on_event("startup")
def startup_event():
    # Billing state is persisted per-user on disk in api.py.
    logger.info("CentaurWeb backend started (billing: per-user persisted)")
    try:
        validate_smtp_config()
        smtp_status = get_smtp_status()
        logger.info(
            f"SMTP status: configured={smtp_status.get('configured')} source={smtp_status.get('source')}"
        )
    except Exception as e:
        logger.error(f"SMTP validation failed: {e}")
        raise
    _auto_commit_on_reload()


@app.on_event("shutdown")
def shutdown_event():
    logger.info("CentaurWeb backend stopped")


# ======================
# API ROUTES (MUST COME FIRST)
# ======================
app.include_router(auth_router, prefix="/api")
app.include_router(api_router, prefix="/api", dependencies=[Depends(require_user)])

# ======================
# FRONTEND
# ======================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Serve /assets directly (so /assets/centaur-logo.png resolves)
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

# Optional: keep /frontend working (useful for direct browsing/debug)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

@app.get("/")
def serve_index():
    return FileResponse(
        str(FRONTEND_DIR / "index.html"),
        headers={"Cache-Control": "no-store"},
    )

# Back-compat: if anything still references /static/*
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)


def _auto_commit_on_reload() -> None:
    """
    Best-effort auto-commit when the server reloads.
    Skips if disabled or if no git changes are present.
    """
    if os.getenv("CENTAUR_AUTO_COMMIT", "1").strip() == "0":
        return
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if inside.returncode != 0:
            return
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not status.stdout.strip():
            return
        subprocess.run(["git", "add", "-A"], check=False)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subprocess.run(["git", "commit", "-m", f"auto: reload {ts}"], check=False)
        subprocess.run(["git", "push"], check=False)
    except Exception as e:
        logger.warning(f"Auto-commit on reload skipped: {e}")
