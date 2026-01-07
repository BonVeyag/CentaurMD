from pathlib import Path

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import router as api_router
from app.auth import router as auth_router, require_user
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
    # Option A: in-memory billing state lives inside api.py (process memory).
    # This confirms startup, but does not persist billing across restarts.
    logger.info("CentaurWeb backend started (billing: in-memory Option A)")
    _auto_commit_on_reload()


@app.on_event("shutdown")
def shutdown_event():
    # Option A behavior: all in-memory state (including any billing day state)
    # will be lost on restart/shutdown.
    logger.info("CentaurWeb backend stopped (billing: in-memory state cleared)")


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
