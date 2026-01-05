# app/attachment_store.py
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, List, Tuple

# IMPORTANT:
# api.py imports these exact names:
#   from app.attachment_store import ATTACHMENT_BLOBS, LOCK
#
# In your current api.py, it expects:
#   ATTACHMENT_BLOBS[attachment_id] = raw_bytes
# not a nested dict by session_id, and not AttachmentRecord objects.
#
# This file keeps that exact contract, while adding robust validation
# for PDF + image uploads (png/jpeg/webp) and a session-scoped metadata index.

LOCK = threading.RLock()

# Structure expected by api.py:
#   ATTACHMENT_BLOBS[attachment_id] = bytes
ATTACHMENT_BLOBS: Dict[str, bytes] = {}


# -------------------------
# Optional metadata index
# -------------------------
# This is NOT required by api.py, but is helpful if you later want to:
# - list attachments by session
# - validate attachment belongs to a session
# - show filename/mime/kind without storing bytes in SessionContext
#
# Nothing in api.py imports this, so it's safe to keep as an internal detail.
_SESSION_INDEX: Dict[str, Dict[str, "AttachmentRecord"]] = {}

# -------------------------
# Validation knobs
# -------------------------
# Keep conservative defaults; adjust as needed.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PDF_BYTES = 25 * 1024 * 1024         # 25 MB
MAX_IMAGE_BYTES = 12 * 1024 * 1024       # 12 MB

_ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}
_ALLOWED_PDF_MIMES = {"application/pdf"}
_ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

# Minimal header sizes for sniffing
_PDF_MAGIC = b"%PDF-"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC_PREFIX = b"\xff\xd8\xff"  # SOI + marker
# WebP: RIFF....WEBP (bytes 0..3 RIFF, bytes 8..11 WEBP)
_WEBP_RIFF = b"RIFF"
_WEBP_WEBP = b"WEBP"


@dataclass
class AttachmentRecord:
    id: str
    session_id: str
    filename: str
    mime: str
    kind: str          # "pdf" | "image"
    size_bytes: int
    created_at: float  # epoch seconds
    sha256: Optional[str] = None  # optional integrity / dedupe support later

    def meta(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Internal helpers
# -------------------------
def _ensure_session_bucket(session_id: str) -> None:
    if session_id not in _SESSION_INDEX:
        _SESSION_INDEX[session_id] = {}


def _sanitize_filename(filename: str) -> str:
    """
    Prevent path traversal and strip control chars.
    Keeps a user-friendly base name.
    """
    fn = (filename or "").strip()
    fn = os.path.basename(fn)  # drops any directory components
    fn = fn.replace("\x00", "")  # null bytes
    fn = re.sub(r"[\r\n\t]+", " ", fn).strip()
    fn = re.sub(r"\s{2,}", " ", fn).strip()
    if not fn:
        fn = "upload"
    # Hard cap (avoid absurd header sizes / UI overflow)
    if len(fn) > 180:
        root, ext = os.path.splitext(fn)
        fn = root[:160] + ext[:20]
    return fn


def _ext_of(filename: str) -> str:
    return os.path.splitext((filename or "").lower().strip())[1]


def _sniff_kind_and_mime(data: bytes) -> Tuple[str, str]:
    """
    Sniff by signature. Returns (kind, mime).
    Supports: PDF, PNG, JPEG, WEBP.
    Raises ValueError if unknown.
    """
    b = data or b""
    if len(b) < 12:
        raise ValueError("File too small to validate")

    # PDF
    if b.startswith(_PDF_MAGIC):
        return "pdf", "application/pdf"

    # PNG
    if b.startswith(_PNG_MAGIC):
        return "image", "image/png"

    # JPEG
    if b.startswith(_JPEG_MAGIC_PREFIX):
        return "image", "image/jpeg"

    # WEBP: RIFF....WEBP
    if b.startswith(_WEBP_RIFF) and b[8:12] == _WEBP_WEBP:
        return "image", "image/webp"

    raise ValueError("Unsupported file type (only PDF, PNG, JPEG, WEBP are allowed)")


def _validate_and_normalize(
    filename: str,
    declared_mime: str,
    data: bytes,
) -> Tuple[str, str, str]:
    """
    Returns (sanitized_filename, normalized_mime, kind).
    Performs size checks + signature checks.
    """
    fn = _sanitize_filename(filename)
    mt_decl = (declared_mime or "").strip().lower()
    b = data or b""
    size = len(b)

    if size <= 0:
        raise ValueError("Empty file")
    if size > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"File too large (max {MAX_ATTACHMENT_BYTES} bytes)")

    ext = _ext_of(fn)
    if ext and ext not in _ALLOWED_EXTS:
        # We still sniff; but this avoids weird uploads like .exe named as image/*
        raise ValueError("Unsupported file extension (only .pdf, .png, .jpg/.jpeg, .webp)")

    # Sniff actual content (source of truth)
    kind, mt_sniff = _sniff_kind_and_mime(b)

    # Enforce size per kind
    if kind == "pdf" and size > MAX_PDF_BYTES:
        raise ValueError(f"PDF too large (max {MAX_PDF_BYTES} bytes)")
    if kind == "image" and size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large (max {MAX_IMAGE_BYTES} bytes)")

    # If declared mime is present and conflicts, ignore it (but keep sniffed).
    # You can optionally log conflicts in api.py if you want.
    normalized_mime = mt_sniff

    # Final allowlist enforcement by normalized mime
    if kind == "pdf" and normalized_mime not in _ALLOWED_PDF_MIMES:
        raise ValueError("Unsupported PDF mime")
    if kind == "image" and normalized_mime not in _ALLOWED_IMAGE_MIMES:
        raise ValueError("Unsupported image mime")

    return fn, normalized_mime, kind


# -------------------------
# Public API (used by api.py indirectly via ATTACHMENT_BLOBS)
# -------------------------
def add_attachment(
    session_id: str,
    filename: str,
    mime: str,
    data: bytes,
    attachment_id: Optional[str] = None,
) -> AttachmentRecord:
    """
    Adds attachment bytes into ATTACHMENT_BLOBS (flat dict, keyed by attachment_id),
    and also records metadata in an internal per-session index.

    Validates: PDF, PNG, JPEG, WEBP only.
    """
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id required")

    aid = (attachment_id or uuid.uuid4().hex).strip()
    if not aid:
        raise ValueError("attachment_id required")

    fn, mt, kind = _validate_and_normalize(filename, mime, data)
    b = data or b""

    sha = hashlib.sha256(b).hexdigest()

    rec = AttachmentRecord(
        id=aid,
        session_id=sid,
        filename=fn,
        mime=mt,
        kind=kind,
        size_bytes=len(b),
        created_at=time.time(),
        sha256=sha,
    )

    with LOCK:
        # bytes store (what api.py expects)
        ATTACHMENT_BLOBS[aid] = b

        # metadata index (optional)
        _ensure_session_bucket(sid)
        _SESSION_INDEX[sid][aid] = rec

    return rec


def list_attachments(session_id: str) -> List[Dict[str, Any]]:
    """
    Lists attachment metadata for a session from the internal index.
    NOTE: Your current api.py may list attachments from context.attachments instead.
    This is here for future use and debugging.
    """
    sid = str(session_id or "").strip()
    with LOCK:
        bucket = _SESSION_INDEX.get(sid, {})
        items = sorted(bucket.values(), key=lambda r: r.created_at)
        return [r.meta() for r in items]


def get_attachment(session_id: str, attachment_id: str) -> Optional[AttachmentRecord]:
    """
    Returns metadata record from the internal index (not bytes).
    """
    sid = str(session_id or "").strip()
    aid = str(attachment_id or "").strip()
    if not sid or not aid:
        return None
    with LOCK:
        return _SESSION_INDEX.get(sid, {}).get(aid)


def get_bytes(attachment_id: str) -> Optional[bytes]:
    """
    Returns raw bytes from the flat bytes store.
    """
    aid = str(attachment_id or "").strip()
    if not aid:
        return None
    with LOCK:
        return ATTACHMENT_BLOBS.get(aid)


def delete_attachment(session_id: str, attachment_id: str) -> bool:
    """
    Deletes from both ATTACHMENT_BLOBS and the internal session index.
    """
    sid = str(session_id or "").strip()
    aid = str(attachment_id or "").strip()
    if not sid or not aid:
        return False

    with LOCK:
        existed = False

        if aid in ATTACHMENT_BLOBS:
            ATTACHMENT_BLOBS.pop(aid, None)
            existed = True

        bucket = _SESSION_INDEX.get(sid)
        if bucket and aid in bucket:
            bucket.pop(aid, None)
            existed = True
            if not bucket:
                _SESSION_INDEX.pop(sid, None)

        return existed


def clear_session_attachments(session_id: str) -> int:
    """
    Clears all attachments for a session (bytes + metadata).
    Returns number of attachments removed (best-effort count).
    """
    sid = str(session_id or "").strip()
    if not sid:
        return 0

    with LOCK:
        bucket = _SESSION_INDEX.pop(sid, None) or {}
        removed_ids = list(bucket.keys())

        count = 0
        for aid in removed_ids:
            if aid in ATTACHMENT_BLOBS:
                ATTACHMENT_BLOBS.pop(aid, None)
                count += 1

        return max(count, len(removed_ids))


# -------------------------
# Optional convenience (not used by api.py)
# -------------------------
def is_supported_upload(filename: str, mime: str, data: bytes) -> bool:
    """
    Quick boolean validator for callers that prefer True/False.
    """
    try:
        _validate_and_normalize(filename, mime, data)
        return True
    except Exception:
        return False