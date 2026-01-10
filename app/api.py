from __future__ import annotations

import base64
import io
import inspect
import json
import logging
import os
import re
import time
from datetime import datetime, date, timezone
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
from threading import Lock as ThreadLock

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from zoneinfo import ZoneInfo

from app.services import (
    make_soap,
    run_differential_coach,
    run_clinical_query,
    run_clinical_query_stream,
    promote_factual_text_to_background,
    generate_referral_letter,
    update_demographics_from_text,
    extract_demographics_from_text,  # <-- unify identifier parsing with services
    generate_patient_summary,
)
from app.transcription import transcribe_audio_bytes
from app.attachment_store import (
    ATTACHMENT_BLOBS,
    LOCK,
    add_attachment as store_add_attachment,
    delete_attachment as store_delete_attachment,
)
from app.auth import require_user, user_billing_dir, AuthUser, send_admin_email
from app.macro_store import (
    list_macros_for_user as store_list_macros_for_user,
    save_macro_for_user as store_save_macro_for_user,
    delete_macro_for_user as store_delete_macro_for_user,
)
from app.models import (
    SessionContext,
    SessionMeta,
    PatientAnchor,
    ClinicalBackground,
    Transcript,
    ClinicianInputs,
    InteractionState,
    DerivedOutputs,
    SoapNoteOutput,
    ReferralLetterOutput,
    AttachmentMeta,
)

# -------------------------
# TEMPORARY in-memory store
# -------------------------
SESSION_STORE: dict[str, SessionContext] = {}

# NOTE: keep router prefixing handled in main.py (include_router(router, prefix="/api"))
router = APIRouter()
logger = logging.getLogger("centaurweb.api")

# =========================
# Billing model + reference
# =========================

BILLING_MODEL_NAME = os.getenv("BILLING_MODEL", "gpt-5.2")
DEFAULT_BILLING_REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "billing_reference.txt",
)
BILLING_REFERENCE_PATH = (os.getenv("CENTAUR_BILLING_REFERENCE_PATH") or DEFAULT_BILLING_REFERENCE_PATH).strip()
BILLING_REFERENCE_MAX_CHARS = 8000
_BILLING_REFERENCE_CACHE: Optional[str] = None
_BILLING_REFERENCE_MTIME: Optional[float] = None

# =========================
# Attachments config
# =========================

ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_PDF_MIMES = {"application/pdf"}
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024  # 15 MB cap

# Keep prompt size sane when injecting attachment-derived text into clinical_query
MAX_ATTACHMENTS_TEXT_CHARS = 12000

# Keep image payloads sane if/when you enable multimodal in services
MAX_IMAGE_BYTES_FOR_MODEL = 2 * 1024 * 1024  # 2 MB per image after downscale/re-encode (best-effort)
MAX_IMAGES_FOR_QUERY = 4

# =========================
# Billing (daily list) store
# =========================

EDMONTON_TZ = ZoneInfo("America/Edmonton")

BILLING_LOCK = ThreadLock()
# Keyed by "{username}:{YYYY-MM-DD}"
DAILY_BILLING_STORE: Dict[str, Dict[str, Any]] = {}
#
# Store shape:
# {
#   "date": "YYYY-MM-DD",
#   "physician": "",
#   "billing_model": "FFS" | "PCPCM",
#   "billing_text": "",
#   "last_updated_at": "ISO",
# }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _now_edmonton() -> datetime:
    return datetime.now(EDMONTON_TZ)


def _today_key_edmonton() -> str:
    return _now_edmonton().date().isoformat()


def _billing_state_key(username: str, day_key: str) -> str:
    return f"{username}:{day_key}"


def _billing_state_path(username: str, day_key: str) -> str:
    base = _billing_archive_dir(username)
    return os.path.join(base, f"current_{day_key}.json")


def _default_billing_state(day_key: str) -> Dict[str, Any]:
    return {
        "date": day_key,
        "physician": "",
        "billing_model": "FFS",
        "billing_text": "",
        "last_updated_at": _utcnow().isoformat(),
    }


def _load_billing_state_from_disk(username: str, day_key: str) -> Dict[str, Any]:
    path = _billing_state_path(username, day_key)
    if not os.path.exists(path):
        return _default_billing_state(day_key)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _default_billing_state(day_key)
    if not isinstance(data, dict):
        return _default_billing_state(day_key)
    data.setdefault("date", day_key)
    data.setdefault("physician", "")
    data.setdefault("billing_model", "FFS")
    data.setdefault("billing_text", "")
    data.setdefault("last_updated_at", _utcnow().isoformat())
    return data


def _persist_billing_state(username: str, st: Dict[str, Any]) -> None:
    day_key = str(st.get("date") or _today_key_edmonton())
    path = _billing_state_path(username, day_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _init_daily_billing_state_if_missing(username: str, day_key: str) -> Dict[str, Any]:
    key = _billing_state_key(username, day_key)
    with BILLING_LOCK:
        st = DAILY_BILLING_STORE.get(key)
        if st is not None:
            return st
    st = _load_billing_state_from_disk(username, day_key)
    with BILLING_LOCK:
        DAILY_BILLING_STORE[key] = st
        return st


def _touch_billing_state(username: str, st: Dict[str, Any]) -> None:
    try:
        st["last_updated_at"] = _utcnow().isoformat()
        _persist_billing_state(username, st)
    except Exception:
        pass


def _billing_archive_dir(username: str) -> str:
    return user_billing_dir(username)


def _safe_billing_archive_path(username: str, filename: str) -> str:
    name = (filename or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not re.match(r"^[A-Za-z0-9._-]+\.txt$", name):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    base = os.path.abspath(_billing_archive_dir(username))
    path = os.path.abspath(os.path.join(base, name))
    if not path.startswith(base + os.sep):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return path


def _make_billing_archive_filename(base_dir: str, dt_local: datetime) -> str:
    stem = dt_local.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{stem}.txt"
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return filename
    idx = 2
    while True:
        filename = f"{stem}_{idx:02d}.txt"
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            return filename
        idx += 1


def _infer_attachment_kind(mime: str) -> str:
    """
    Backward-compatible helper.
    """
    m = (mime or "").lower().strip()
    if m in ALLOWED_PDF_MIMES:
        return "pdf"
    if m in ALLOWED_IMAGE_MIMES:
        return "image"
    return "unknown"


def _sniff_mime_from_bytes(data: bytes) -> Optional[str]:
    """
    Best-effort MIME sniffing to prevent content_type spoofing.
    Returns one of the allowlisted mimes or None if unknown.
    """
    if not data:
        return None

    b = data[:32]

    # PDF: %PDF-
    if b.startswith(b"%PDF-"):
        return "application/pdf"

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    # JPEG: FF D8 FF
    if len(b) >= 3 and b[0:3] == b"\xFF\xD8\xFF":
        return "image/jpeg"

    # WEBP: "RIFF" .... "WEBP"
    if len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"

    return None


# =========================
# Helpers
# =========================

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()


def _maybe_parse_json_string(x):
    """
    If x is a JSON-looking string OR code-fenced JSON string, try json.loads.
    Also attempts to extract the first {...} object if mixed with other text.
    Returns parsed object if successful, else returns the original input.
    """
    if not isinstance(x, str):
        return x

    s = _strip_code_fences(x)
    if not s:
        return x

    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    if "{" in s and "}" in s:
        candidate = s[s.find("{"):s.rfind("}") + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return x

    return x


def _safe_cache_derived(context: SessionContext, field: str, value) -> None:
    """
    Avoid crashing if DerivedOutputs doesn't have the field yet (extra=forbid)
    or if validate_assignment rejects type. Best-effort cache only.
    """
    try:
        setattr(context.derived_outputs, field, value)
    except Exception as e:
        logger.warning(f"DerivedOutputs cache skipped for field '{field}': {e}")


def _get_context_or_404(session_id: str) -> SessionContext:
    sid = (session_id or "").strip()
    if not sid or sid not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSION_STORE[sid]


def _touch(context: SessionContext) -> None:
    try:
        context.session_meta.last_updated_at = _utcnow()
    except Exception:
        pass


def _clip_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    head = max_chars // 2
    tail = max_chars - head
    return t[:head].rstrip() + "\n\n[...clipped...]\n\n" + t[-tail:].lstrip()


def _fn_accepts_kw(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        if name in sig.parameters:
            return True
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
    except Exception:
        pass
    return False


def _clean_digits(s: str) -> str:
    return re.sub(r"\D+", "", (s or "").strip())


def _clean_phn(phn: str) -> str:
    """
    Alberta PHN cleaning (safe; prevents phone-number truncation mistakes).

    - Prefer exactly 9 digits.
    - Allow leading-zero padding (e.g., 00 + 9 digits) by stripping leading zeros to 9 digits.
    - Reject arbitrary >9 digit strings (do NOT take last 9 digits) unless the prefix is all zeros.
    - If under-length but plausible (>=7 digits), return digits (best-effort); otherwise "".
    """
    raw = (phn or "").strip()
    digits = re.sub(r"\D+", "", raw)

    if len(digits) == 9:
        return digits

    if len(digits) > 9:
        prefix = digits[:-9]
        if prefix and set(prefix) <= {"0"}:
            return digits[-9:]
        return ""

    if len(digits) >= 7:
        return digits

    return ""


def _get_patient_obj(context: SessionContext):
    """
    Compatibility helper:
    - Prefer context.patient_anchor
    - Fall back to context.patient if older code existed
    """
    if hasattr(context, "patient_anchor"):
        pa = getattr(context, "patient_anchor", None)
        if pa is not None:
            return pa
    if hasattr(context, "patient"):
        p = getattr(context, "patient", None)
        if p is not None:
            return p
    return None


def _reset_patient_identifiers(context: SessionContext) -> None:
    """
    Critical fix for “stale identifiers across patients in the same session”.

    If the frontend reuses a session_id across different patients and simply posts a new
    /clinical_background EMR block, prior identifiers (name/phn/age/etc) can persist.
    """
    # Preferred: replace the PatientAnchor object entirely
    try:
        if hasattr(context, "patient_anchor"):
            context.patient_anchor = PatientAnchor()
            return
    except Exception:
        pass

    # Fallback: blank the fields on the existing anchor/patient object
    pa = _get_patient_obj(context)
    if pa is None:
        return

    for attr, blank in (
        ("name", ""),
        ("phn", ""),
        ("dob", ""),
        ("sex", ""),
        ("gender", ""),
        ("age", None),
    ):
        try:
            setattr(pa, attr, blank)
        except Exception:
            continue


def _ensure_anchor_hydrated_from_emr(context: SessionContext) -> None:
    """
    Policy + determinism:
    - Hydrate identifiers from EMR/background only.
    - Use overwrite=True when supported so EMR remains source-of-truth even if a reset is missed.
    """
    try:
        emr = (context.clinical_background.emr_dump or "").strip()
    except Exception:
        emr = ""
    if not emr:
        return

    try:
        if _fn_accepts_kw(update_demographics_from_text, "overwrite"):
            update_demographics_from_text(context, emr, overwrite=True)
        else:
            update_demographics_from_text(context, emr)
    except Exception:
        pass


# =========================
# Attachment extraction (PDF -> text; Image -> payload if supported)
# =========================

def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Best-effort PDF text extraction.
    - Works for text-based PDFs.
    - Scanned PDFs may return empty text (no OCR here).
    """
    if not data:
        return ""

    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return ""

    try:
        reader = PdfReader(io.BytesIO(data))
        parts: List[str] = []
        for page in getattr(reader, "pages", []) or []:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = txt.strip()
            if txt:
                parts.append(txt)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def _collect_attachments_text_for_query(context: SessionContext) -> str:
    """
    Builds a single text block suitable for injecting into /clinical_query prompts.
    Includes per-file headers and clips to MAX_ATTACHMENTS_TEXT_CHARS.
    """
    metas = context.attachments or []
    if not metas:
        return ""

    blocks: List[str] = []

    for meta in metas:
        if not meta or not getattr(meta, "id", None):
            continue

        att_id = meta.id
        with LOCK:
            blob = ATTACHMENT_BLOBS.get(att_id)

        if not blob:
            continue

        kind = (getattr(meta, "kind", "") or "").lower().strip()
        filename = (getattr(meta, "filename", "") or "").strip() or att_id

        if kind != "pdf":
            continue

        extracted = _extract_text_from_pdf_bytes(blob)
        extracted = (extracted or "").strip()

        if not extracted:
            blocks.append(f"=== {filename} (pdf) ===\n[No extractable text found]\n")
        else:
            blocks.append(f"=== {filename} (pdf) ===\n{extracted}\n")

    joined = "\n".join([b.strip() for b in blocks if b.strip()]).strip()
    return _clip_text(joined, MAX_ATTACHMENTS_TEXT_CHARS)


def _downscale_image_best_effort(data: bytes, mime: str) -> tuple[bytes, str]:
    """
    Attempts to downscale/re-encode an image so it is safe to pass to a model.
    Returns (bytes, mime). If Pillow is unavailable or fails, returns original.
    """
    if not data:
        return b"", mime or "application/octet-stream"

    if len(data) <= MAX_IMAGE_BYTES_FOR_MODEL:
        return data, mime or "application/octet-stream"

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return data, mime or "application/octet-stream"

    try:
        im = Image.open(io.BytesIO(data))
        im.load()

        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")

        max_side = 1400
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))

        out = io.BytesIO()
        im.save(out, format="JPEG", quality=75, optimize=True)
        out_bytes = out.getvalue()

        tries = 0
        while len(out_bytes) > MAX_IMAGE_BYTES_FOR_MODEL and tries < 3:
            tries += 1
            w2, h2 = im.size
            im = im.resize((max(1, int(w2 * 0.8)), max(1, int(h2 * 0.8))))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=max(55, 75 - tries * 10), optimize=True)
            out_bytes = out.getvalue()

        return out_bytes, "image/jpeg"
    except Exception:
        return data, mime or "application/octet-stream"


def _collect_image_payload_for_query(context: SessionContext) -> List[Dict[str, Any]]:
    """
    Returns a list of image payload dicts (best-effort) for multimodal services.
    This does NOT change behavior unless app.services.run_clinical_query accepts
    an 'attachments' kwarg (or **kwargs).
    """
    metas = context.attachments or []
    if not metas:
        return []

    payloads: List[Dict[str, Any]] = []
    for meta in metas:
        if len(payloads) >= MAX_IMAGES_FOR_QUERY:
            break

        if not meta or not getattr(meta, "id", None):
            continue

        kind = (getattr(meta, "kind", "") or "").lower().strip()
        if kind != "image":
            continue

        att_id = meta.id
        filename = (getattr(meta, "filename", "") or "").strip() or att_id
        mime = (getattr(meta, "mime", "") or "").lower().strip() or "application/octet-stream"

        with LOCK:
            blob = ATTACHMENT_BLOBS.get(att_id)

        if not blob:
            continue

        blob2, mime2 = _downscale_image_best_effort(blob, mime)
        if not blob2:
            continue

        b64 = base64.b64encode(blob2).decode("ascii")
        payloads.append(
            {
                "id": att_id,
                "filename": filename,
                "mime": mime2,
                "kind": "image",
                "data_base64": b64,
            }
        )

    return payloads


def _augment_query_with_attachments_text(query: str, attachments_text: str) -> str:
    """
    If services does NOT accept attachment kwargs, we inject extracted PDF text
    directly into the user prompt in a bounded, explicit section.
    """
    at = (attachments_text or "").strip()
    if not at:
        return query

    return (
        f"{query.rstrip()}\n\n"
        f"---\n"
        f"ATTACHMENTS (EXTRACTED TEXT)\n"
        f"Use the following extracted text from attached documents as part of the conversation context.\n"
        f"---\n"
        f"{at}\n"
        f"---\n"
    ).strip()


def _merge_macro_and_query(query: str, macro: str) -> str:
    """
    Combine macro instructions with user query for model context.
    """
    q = (query or "").strip()
    m = (macro or "").strip()
    if m and q:
        return f"{m}\n\n---\nUSER QUESTION:\n{q}".strip()
    if m:
        return m
    return q


def format_differential_output(output) -> str:
    if output is None:
        return "— no differential —"

    output = _maybe_parse_json_string(output)

    if isinstance(output, str):
        return output.strip()

    if isinstance(output, dict):
        ddx = output.get("ddx") or []
        cant_miss = output.get("cant_miss_questions") or []
        key_q = output.get("key_questions_to_refine_ddx") or []
        plan = output.get("suggested_plan") or []
        confidence = (output.get("confidence") or "").strip()

        lines: list[str] = []

        if ddx:
            lines.append("Ddx")
            for i, x in enumerate(ddx[:10], start=1):
                s = str(x).strip()
                if s:
                    lines.append(f"{i}. {s}")

        if cant_miss:
            lines.append("")
            lines.append("Can't miss")
            for x in cant_miss[:10]:
                s = str(x).strip()
                if s:
                    lines.append(f"- {s}")

        if key_q:
            lines.append("")
            lines.append("Key questions to refine ddx")
            for x in key_q[:12]:
                s = str(x).strip()
                if s:
                    lines.append(f"- {s}")

        if plan:
            lines.append("")
            lines.append("Suggested plan")
            for x in plan[:12]:
                s = str(x).strip()
                if s:
                    lines.append(f"- {s}")

        if confidence:
            lines.append("")
            lines.append(f"Confidence: {confidence}")

        return "\n".join(lines).strip() if lines else "— no differential content —"

    try:
        return json.dumps(output, indent=2, ensure_ascii=False)
    except Exception:
        return str(output)


def _age_from_dob(dob: date) -> int:
    today = _now_edmonton().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


# =========================
# Health check
# =========================

@router.get("/ping")
def ping():
    return {"message": "CentaurWeb is alive"}


# =========================
# Audio transcription
# =========================

@router.post("/transcribe_chunk")
async def transcribe_chunk(session_id: str, file: UploadFile = File(...)):
    context = _get_context_or_404(session_id)

    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 3000:
        return {"text": ""}

    start = time.time()
    try:
        text = transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            filename=file.filename or "chunk.webm"
        )

        if not getattr(context.transcript, "raw_text", None):
            context.transcript.raw_text = ""

        context.transcript.raw_text = (context.transcript.raw_text + " " + (text or "")).strip()

        # IMPORTANT: Do NOT hydrate identifiers from transcript (EMR-only invariant).
        _touch(context)

        elapsed = round(time.time() - start, 2)
        logger.info(f"Chunk transcribed in {elapsed}s (sid={context.session_meta.session_id})")

        return {"text": (text or "").strip()}

    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# =========================
# Session lifecycle
# =========================

@router.post("/session/create")
def create_session():
    now = _utcnow()
    session_id = str(uuid4())

    context = SessionContext(
        session_meta=SessionMeta(
            session_id=session_id,
            created_at=now,
            last_updated_at=now,
        ),
        patient_anchor=PatientAnchor(),
        clinical_background=ClinicalBackground(),
        transcript=Transcript(),
        clinician_inputs=ClinicianInputs(),
        interaction_state=InteractionState(),
        derived_outputs=DerivedOutputs(),
    )

    try:
        context.attachments = []
    except Exception:
        pass

    SESSION_STORE[session_id] = context
    return {"session_id": session_id}


# =========================
# Attachments (upload / list / delete)
# =========================

@router.post("/session/{session_id}/attachments")
async def upload_attachment(session_id: str, file: UploadFile = File(...)):
    """
    Upload PDF or image.

    Important change:
    - We validate by sniffing bytes (best-effort), not trusting content_type.
    """
    context = _get_context_or_404(session_id)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_ATTACHMENT_BYTES} bytes)")

    declared_mime = (file.content_type or "").lower().strip()
    sniffed_mime = _sniff_mime_from_bytes(data)
    mime = sniffed_mime or declared_mime

    kind = _infer_attachment_kind(mime)
    if kind == "unknown":
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime or 'unknown'}")

    try:
        rec = store_add_attachment(
            session_id=session_id,
            filename=file.filename or "upload",
            mime=mime,
            data=data,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Attachment upload failed")
        raise HTTPException(status_code=500, detail=f"Attachment upload failed: {str(e)}")

    if (rec.kind or "").lower().strip() not in ("pdf", "image"):
        with LOCK:
            ATTACHMENT_BLOBS.pop(rec.id, None)
        raise HTTPException(status_code=400, detail=f"Unsupported file kind: {rec.kind}")

    meta = AttachmentMeta(
        id=rec.id,
        kind=rec.kind,
        filename=rec.filename,
        mime=rec.mime,
        size_bytes=rec.size_bytes,
        uploaded_at=_utcnow().isoformat(),
    )

    context.attachments = (context.attachments or []) + [meta]
    _touch(context)

    if hasattr(meta, "model_dump"):
        return {"attachment": meta.model_dump()}
    return {"attachment": meta.dict()}  # pydantic v1 fallback


@router.get("/session/{session_id}/attachments")
def list_attachments(session_id: str):
    context = _get_context_or_404(session_id)
    items = context.attachments or []
    out = []
    for m in items:
        if hasattr(m, "model_dump"):
            out.append(m.model_dump())
        else:
            out.append(m.dict())
    return {"attachments": out}


@router.delete("/session/{session_id}/attachments/{attachment_id}")
def delete_attachment(session_id: str, attachment_id: str):
    context = _get_context_or_404(session_id)

    context.attachments = [a for a in (context.attachments or []) if a.id != attachment_id]

    try:
        store_delete_attachment(session_id=session_id, attachment_id=attachment_id)
    except Exception:
        with LOCK:
            ATTACHMENT_BLOBS.pop(attachment_id, None)

    _touch(context)
    return {"ok": True}


# =========================
# MAKE SOAP
# =========================

@router.post("/session/{session_id}/make_soap")
def make_soap_endpoint(session_id: str):
    context = _get_context_or_404(session_id)

    _ensure_anchor_hydrated_from_emr(context)
    result = make_soap(context)

    try:
        soap_obj = SoapNoteOutput(
            text=result["soap_text"],
            structured=None,
            generated_at=result["generated_at"],
        )
        _safe_cache_derived(context, "soap_note", soap_obj)
    except Exception as e:
        logger.warning(f"SOAP cache skipped: {e}")

    _touch(context)
    return {"soap": result["soap_text"]}


# =========================
# DIFFERENTIAL COACH
# =========================

@router.post("/session/{session_id}/differential")
def differential_coach_endpoint(session_id: str):
    context = _get_context_or_404(session_id)

    _ensure_anchor_hydrated_from_emr(context)
    output_raw = run_differential_coach(context)
    output_text = output_raw

    _safe_cache_derived(context, "differential", output_text)
    _touch(context)

    return {"differential": output_text, "generated_at": _utcnow().isoformat()}


# =========================
# Clinical Background (manual)
# =========================

class ClinicalBackgroundPayload(BaseModel):
    background_text: str


@router.post("/session/{session_id}/clinical_background")
def update_clinical_background(session_id: str, payload: ClinicalBackgroundPayload):
    """
    IMPORTANT:
    - Always hydrate PatientAnchor identifiers from this EMR text.
    - CRITICAL FIX: reset identifiers first so new-patient EMR overwrites stale prior-patient values.
    - Do not rely on transcript for identifiers.
    """
    context = _get_context_or_404(session_id)

    bg = (payload.background_text or "").strip()
    context.clinical_background.emr_dump = bg

    _reset_patient_identifiers(context)
    _ensure_anchor_hydrated_from_emr(context)
    _touch(context)

    logger.info(f"Clinical background updated for session {session_id}")
    return {"status": "ok"}


# =========================
# Referral Letter (plain text)
# =========================

@router.post("/session/{session_id}/referral")
def referral_letter(session_id: str):
    context = _get_context_or_404(session_id)

    _ensure_anchor_hydrated_from_emr(context)
    letter_text = generate_referral_letter(context)
    now_iso = _utcnow().isoformat()

    try:
        existing = getattr(context.derived_outputs, "referrals", None)
        if not isinstance(existing, list):
            existing = []
        existing.append(
            ReferralLetterOutput(
                id=str(uuid4()),
                text=letter_text,
                generated_at=now_iso,
            )
        )
        _safe_cache_derived(context, "referrals", existing)
    except Exception as e:
        logger.warning(f"Referral cache skipped: {e}")

    _touch(context)
    return {"referral": letter_text, "generated_at": now_iso}


# =========================
# Macro Store
# =========================

class MacroPayload(BaseModel):
    name: str
    content: str


@router.get("/macros")
def list_macros(user: AuthUser = Depends(require_user)):
    return {"macros": store_list_macros_for_user(user.username)}


@router.post("/macros")
def create_macro(payload: MacroPayload, user: AuthUser = Depends(require_user)):
    name = (payload.name or "").strip()
    content = (payload.content or "").strip()
    if not name or not content:
        raise HTTPException(status_code=400, detail="Macro name and content are required")
    try:
        macro = store_save_macro_for_user(user.username, None, name, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"macro": macro}


@router.put("/macros/{macro_id}")
def update_macro(macro_id: str, payload: MacroPayload, user: AuthUser = Depends(require_user)):
    name = (payload.name or "").strip()
    content = (payload.content or "").strip()
    if not macro_id:
        raise HTTPException(status_code=400, detail="Macro id is required")
    if not name or not content:
        raise HTTPException(status_code=400, detail="Macro name and content are required")
    try:
        macro = store_save_macro_for_user(user.username, macro_id, name, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"macro": macro}


@router.delete("/macros/{macro_id}")
def delete_macro(macro_id: str, user: AuthUser = Depends(require_user)):
    if not macro_id:
        raise HTTPException(status_code=400, detail="Macro id is required")
    try:
        ok = store_delete_macro_for_user(user.username, macro_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="Macro not found")
    return {"deleted": True}


# =========================
# Clinical Query Console
# =========================

class ClinicalQueryPayload(BaseModel):
    query: str
    mode: str = "fast"
    macro: Optional[str] = None
    model: Optional[str] = None


@router.post("/session/{session_id}/clinical_query")
def clinical_query(session_id: str, payload: ClinicalQueryPayload):
    """
    GPT-like clinical query with persistent factual memory.

    Attachment behavior (backward-compatible):
    - PDFs: extracted text is either
        (a) passed to run_clinical_query as attachments_text (if supported), OR
        (b) injected into the query prompt (fallback).
    - Images: if run_clinical_query supports multimodal via an 'attachments' kwarg,
        we pass base64 image payloads.
    """
    context = _get_context_or_404(session_id)

    _ensure_anchor_hydrated_from_emr(context)

    query_raw = (payload.query or "").strip()
    macro = (payload.macro or "").strip()
    if not query_raw and not macro:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    query = _merge_macro_and_query(query_raw, macro)

    promoted = False
    if query_raw:
        try:
            promoted = promote_factual_text_to_background(context, query_raw)
            if promoted:
                logger.info(f"Promoted factual text into EMR background for {session_id}")
        except Exception as e:
            logger.warning(f"Promotion failed for {session_id}: {e}")

    attachments_text = ""
    try:
        attachments_text = _collect_attachments_text_for_query(context)
    except Exception as e:
        logger.warning(f"Attachment text extraction failed for {session_id}: {e}")
        attachments_text = ""

    image_payloads: List[Dict[str, Any]] = []
    try:
        image_payloads = _collect_image_payload_for_query(context)
    except Exception as e:
        logger.warning(f"Image payload prep failed for {session_id}: {e}")
        image_payloads = []

    call_kwargs: Dict[str, Any] = {"query": query, "mode": payload.mode}
    if payload.model:
        call_kwargs["model_override"] = payload.model
    attachments_used = False

    if attachments_text.strip() and _fn_accepts_kw(run_clinical_query, "attachments_text"):
        call_kwargs["attachments_text"] = attachments_text
        attachments_used = True

    if image_payloads and _fn_accepts_kw(run_clinical_query, "attachments"):
        call_kwargs["attachments"] = image_payloads
        attachments_used = True

    if attachments_text.strip() and "attachments_text" not in call_kwargs and "attachments" not in call_kwargs:
        call_kwargs["query"] = _augment_query_with_attachments_text(query, attachments_text)
        attachments_used = True

    output = run_clinical_query(context, **call_kwargs)
    _touch(context)

    return {
        "response": output,
        "generated_at": _utcnow().isoformat(),
        "promoted_to_background": promoted,
        "attachments_used": attachments_used,
        "attachments_images_sent": bool(image_payloads) and ("attachments" in call_kwargs),
        "attachments_pdf_text_included": bool(attachments_text.strip()),
    }


# =========================
# Clinical Query (Streaming)
# =========================

@router.post("/session/{session_id}/clinical_query_stream")
def clinical_query_stream(session_id: str, payload: ClinicalQueryPayload):
    """
    Streaming clinical query response (NDJSON).
    Each line is a JSON object with:
      - {"type":"chunk","data":"..."} for content
      - {"type":"meta","data":{...}} at end
      - {"type":"error","data":"..."} on failure
    """
    context = _get_context_or_404(session_id)
    _ensure_anchor_hydrated_from_emr(context)

    query_raw = (payload.query or "").strip()
    macro = (payload.macro or "").strip()
    if not query_raw and not macro:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    query = _merge_macro_and_query(query_raw, macro)

    promoted = False
    if query_raw:
        try:
            promoted = promote_factual_text_to_background(context, query_raw)
            if promoted:
                logger.info(f"Promoted factual text into EMR background for {session_id}")
        except Exception as e:
            logger.warning(f"Promotion failed for {session_id}: {e}")

    attachments_text = ""
    try:
        attachments_text = _collect_attachments_text_for_query(context)
    except Exception as e:
        logger.warning(f"Attachment text extraction failed for {session_id}: {e}")
        attachments_text = ""

    image_payloads: List[Dict[str, Any]] = []
    try:
        image_payloads = _collect_image_payload_for_query(context)
    except Exception as e:
        logger.warning(f"Image payload prep failed for {session_id}: {e}")
        image_payloads = []

    call_kwargs: Dict[str, Any] = {"query": query, "mode": payload.mode}
    if payload.model:
        call_kwargs["model_override"] = payload.model
    attachments_used = False

    if attachments_text.strip():
        call_kwargs["attachments_text"] = attachments_text
        attachments_used = True

    if image_payloads:
        call_kwargs["attachments"] = image_payloads
        attachments_used = True

    if attachments_text.strip() and "attachments_text" not in call_kwargs:
        call_kwargs["query"] = _augment_query_with_attachments_text(query, attachments_text)
        attachments_used = True

    _touch(context)

    def _stream():
        try:
            for chunk in run_clinical_query_stream(context, **call_kwargs):
                if chunk:
                    yield json.dumps({"type": "chunk", "data": chunk}) + "\n"
            meta = {
                "generated_at": _utcnow().isoformat(),
                "promoted_to_background": promoted,
                "attachments_used": attachments_used,
                "attachments_images_sent": bool(image_payloads),
                "attachments_pdf_text_included": bool(attachments_text.strip()),
            }
            yield json.dumps({"type": "meta", "data": meta}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


# =========================
# Patient Summary
# =========================

@router.post("/session/{session_id}/patient_summary")
def patient_summary(session_id: str):
    context = _get_context_or_404(session_id)
    _ensure_anchor_hydrated_from_emr(context)

    try:
        summary = generate_patient_summary(context)
    except Exception as e:
        logger.exception("Patient summary failed")
        raise HTTPException(status_code=500, detail=f"Patient summary failed: {str(e)}")

    _touch(context)
    return {"summary": summary}


# =========================
# Billing helpers
# =========================

def _parse_date_any(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    fmts = (
        "%Y-%m-%d", "%Y/%m/%d",
        "%d-%m-%Y", "%d/%m/%Y",
        "%m/%d/%Y", "%m-%d-%Y",
        "%d-%b-%Y", "%d-%B-%Y",
        "%d/%b/%Y", "%d/%B/%Y",
        "%b %d %Y", "%b %d, %Y",
        "%B %d %Y", "%B %d, %Y",
        "%d %b %Y", "%d %B %Y",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    s2 = re.sub(r"\s+", " ", s)
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s2, fmt).date()
        except Exception:
            continue
    return None


def _age_display_years_or_months(age_years: Optional[int], dob_str: str) -> str:
    """
    - If <2 years and DOB parseable: show months.
    - Else: show years.
    """
    if age_years is None and dob_str:
        dob = _parse_date_any(dob_str)
        if dob:
            age_years = _age_from_dob(dob)

    if age_years is None:
        return ""

    if age_years < 2:
        dob = _parse_date_any(dob_str) if dob_str else None
        if dob:
            today = _now_edmonton().date()
            months = (today.year - dob.year) * 12 + (today.month - dob.month)
            if today.day < dob.day:
                months -= 1
            months = max(0, months)
            return f"{months} months" if months != 1 else "1 month"
        # fallback if DOB missing
        return f"{age_years} years"

    return f"{age_years} years" if age_years != 1 else "1 year"


def _patient_line_from_emr(context: SessionContext) -> Tuple[str, Dict[str, Any]]:
    """
    Line 1: "Name | PHN | age_display"
    Must be extracted from EMR dump (not transcript).
    """
    emr = ""
    try:
        emr = (context.clinical_background.emr_dump or "").strip()
    except Exception:
        emr = ""

    name, phn, dob, age, _sex = extract_demographics_from_text(emr)
    phn_clean = _clean_phn(phn or "")
    age_disp = _age_display_years_or_months(age, dob or "")

    name_out = (name or "").strip()
    phn_out = phn_clean if len(phn_clean) == 9 else (phn_clean or "").strip()

    # UI-friendly: avoid totally blank fields
    name_ui = name_out if name_out else "—"
    phn_ui = phn_out if phn_out else "—"
    age_ui = age_disp if age_disp else "—"

    return f"{name_ui} | {phn_ui} | {age_ui}", {"name": name_out, "phn": phn_out, "dob": dob or "", "age_years": age}


def _has_meaningful_transcript_for_billing(context: SessionContext) -> bool:
    t = (getattr(context.transcript, "raw_text", None) or "").strip()
    if len(t) < 60:
        return False
    if len(t.split()) < 12:
        return False
    return True


def _parse_emr_date_line(line: str) -> Optional[date]:
    m = re.match(r"^\s*(\d{4})[-/](\d{1,2}|[A-Za-z]{3,9})[-/](\d{1,2})\b", line or "")
    if not m:
        return None
    try:
        year = int(m.group(1))
        mon_raw = (m.group(2) or "").strip()
        day = int(m.group(3))
    except Exception:
        return None

    if mon_raw.isdigit():
        try:
            return date(year, int(mon_raw), day)
        except Exception:
            return None

    mon_key = mon_raw.strip().lower()[:3]
    mon_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    mon = mon_map.get(mon_key)
    if not mon:
        return None
    try:
        return date(year, mon, day)
    except Exception:
        return None


def _extract_last_dated_emr_entry(context: SessionContext) -> str:
    """
    Best-effort extraction of the most recent dated EMR entry block.
    Looks for lines like "2025-Dec-30" and captures until the next date line.
    """
    emr = (getattr(context.clinical_background, "emr_dump", None) or "").strip()
    if not emr:
        return ""

    lines = emr.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    date_re = re.compile(r"^\s*(\d{4})[-/](\d{1,2}|[A-Za-z]{3,9})[-/](\d{1,2})\b")

    entries: List[Tuple[Optional[date], int, List[str]]] = []
    current_start: Optional[int] = None
    current_date: Optional[date] = None
    current_lines: List[str] = []

    for i, ln in enumerate(lines):
        if date_re.match(ln):
            if current_start is not None and current_lines:
                entries.append((current_date, current_start, current_lines))
            current_start = i
            current_date = _parse_emr_date_line(ln)
            current_lines = [ln]
            continue
        if current_start is not None:
            current_lines.append(ln)

    if current_start is not None and current_lines:
        entries.append((current_date, current_start, current_lines))

    if not entries:
        return ""

    # Prefer the most recent date by value; fallback to last in text if parsing fails.
    best = None
    for dt, idx, block_lines in entries:
        if dt is None:
            continue
        if best is None or dt > best[0]:
            best = (dt, idx, block_lines)

    chosen_lines = best[2] if best else entries[-1][2]
    return "\n".join(chosen_lines).strip()


def _extract_icd9_parts(icd_items: Any) -> List[str]:
    parts: List[str] = []
    if not isinstance(icd_items, list):
        return parts

    for it in icd_items[:6]:
        code = ""
        dx = ""
        if isinstance(it, dict):
            code = str(it.get("code") or it.get("icd9") or it.get("icd_9") or "").strip()
            dx = str(it.get("dx") or it.get("description") or it.get("label") or "").strip()
        elif isinstance(it, str):
            s = it.strip()
            m = re.match(r"^\s*([0-9]{3,5}(?:\.\d{1,2})?)\s*[:\-]?\s*(.*)$", s)
            if m:
                code = m.group(1)
                dx = (m.group(2) or "").strip()
        if not code:
            continue
        parts.append(f"{code} ({dx})" if dx else code)

    return parts


def _extract_icd9_from_text_direct(text: str, max_items: int = 4) -> List[str]:
    """
    Deterministic ICD-9 extraction from raw text (best-effort).
    Looks for "Diagnosis 493" or "493 (Diagnosis)" patterns line-by-line.
    """
    if not (text or "").strip():
        return []

    desc_before_code = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9/\-\s&]{2,80}?)\s+(\d{3}(?:\.\d{1,2})?)\b"
    )
    code_before_desc = re.compile(
        r"\b(\d{3}(?:\.\d{1,2})?)\b\s*[\(\-:\s]+([A-Za-z][A-Za-z0-9/\-\s&]{2,80})"
    )

    stopwords = {"author", "date", "signature", "clinic", "dr", "note", "notes"}
    deny_desc_re = re.compile(
        r"\b(kg|lbs|lb|cm|mm|mmhg|bpm|%|bp|weight|height|bmi|pulse|temp|temperature)\b",
        flags=re.IGNORECASE,
    )
    found: List[str] = []
    seen_codes: set[str] = set()

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if any(sw in low for sw in stopwords):
            continue

        m = desc_before_code.match(line)
        if not m:
            m = code_before_desc.search(line)
            if not m:
                continue
            code = m.group(1)
            desc = m.group(2)
        else:
            desc = m.group(1)
            code = m.group(2)

        code = (code or "").strip()
        desc = re.sub(r"\s+", " ", (desc or "").strip())
        desc = re.sub(r"[^\w\s/\-&]", "", desc).strip()
        if deny_desc_re.search(desc):
            continue
        if not code or not desc:
            continue
        if code in seen_codes:
            continue

        found.append(f"{code} ({desc})")
        seen_codes.add(code)
        if len(found) >= max_items:
            break

    return found


def _fetch_icd9_parts_from_source(source_text: str, source_label: str, ref_text: str) -> List[str]:
    if not (source_text or "").strip():
        return []

    ref_block = f"\nREFERENCE (use as guidance, not as a source of diagnoses):\n{ref_text}\n" if ref_text else ""
    prompt = f"""
You are an Alberta (Canada) primary care billing assistant.

TASK:
Extract likely ICD-9 codes (max 4) strictly for diagnoses discussed in the provided source.

HARD RULES:
- Use ONLY the provided source text; do NOT infer from other background.
- If the source does not support a diagnosis, omit it.
- If any diagnosis is mentioned, include at least one ICD-9 code.
- Output MUST be STRICT JSON only (no markdown, no extra text).

OUTPUT JSON SCHEMA:
{{
  "icd9": [{{"code":"401","dx":"Hypertension"}}]
}}

{source_label}:
{source_text}
{ref_block}
""".strip()

    try:
        client = _openai_client_best_effort()
        resp = client.chat.completions.create(
            model=BILLING_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return strict JSON only. Conservative, source-only extraction."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(_strip_code_fences(raw))
        return _extract_icd9_parts(data.get("icd9") or [])
    except Exception:
        return []


def _openai_client_best_effort():
    """
    Local OpenAI client for billing extraction. Keeps api.py self-contained.
    If OPENAI_API_KEY is missing, this will raise and we will return blanks.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK missing: {e}")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _load_billing_reference_text() -> str:
    """
    Optional local reference file to guide billing suggestions.
    Controlled by CENTAUR_BILLING_REFERENCE_PATH.
    """
    global _BILLING_REFERENCE_CACHE, _BILLING_REFERENCE_MTIME

    path = BILLING_REFERENCE_PATH
    if not path:
        return ""

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return ""

    if _BILLING_REFERENCE_CACHE is not None and _BILLING_REFERENCE_MTIME == mtime:
        return _BILLING_REFERENCE_CACHE

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return ""

    text = (text or "").strip()
    if len(text) > BILLING_REFERENCE_MAX_CHARS:
        text = text[:BILLING_REFERENCE_MAX_CHARS].rstrip() + "\n[...reference clipped...]"

    _BILLING_REFERENCE_CACHE = text
    _BILLING_REFERENCE_MTIME = mtime
    return text


_CMGP_RE = re.compile(r"^CMGP(0[1-9]|10)\b", flags=re.IGNORECASE)


def _strip_cmgp_modifiers(one_line: str) -> str:
    """
    Remove CMGP01-CMGP10 from a billing one-line string.
    """
    s = (one_line or "").strip()
    if not s:
        return ""

    if "+" in s:
        parts = [p.strip() for p in s.split("+") if p.strip()]
        kept = [p for p in parts if not _CMGP_RE.match(p)]
        return " + ".join(kept).strip()

    # Fallback: remove inline CMGP tokens and clean up whitespace
    s2 = re.sub(r"\bCMGP(0[1-9]|10)\b", "", s, flags=re.IGNORECASE)
    s2 = re.sub(r"\s{2,}", " ", s2).strip()
    return s2


def _generate_icd9_and_billing_lines(
    context: SessionContext,
    billing_model: str,
) -> Tuple[str, str]:
    """
    Returns:
      line2: ICD-9 line (always includes label; blank if no transcript)
      line3: Billing line (or "")
    Rules:
      - ICD-9 strictly from transcript.
      - If transcript insufficient: return blank ICD-9 label and use EMR fallback only for billing line (if available).
    """
    transcript = (getattr(context.transcript, "raw_text", None) or "").strip()
    transcript_ok = _has_meaningful_transcript_for_billing(context)

    fallback_source = ""
    if not transcript_ok:
        fallback_source = _extract_last_dated_emr_entry(context)
        if not fallback_source:
            return "ICD-9: ", ""

    source_text = transcript if transcript_ok else fallback_source
    source_clip = _clip_text(source_text, 5200)

    model_norm = (billing_model or "FFS").strip().upper()
    if model_norm not in ("FFS", "PCPCM"):
        model_norm = "FFS"

    # Business rule: Raj adds 5 min chart review + 5 min charting for FFS
    overhead_minutes = 10 if model_norm == "FFS" else 0
    ref_text = _load_billing_reference_text()
    ref_block = f"\nREFERENCE (use as guidance, not as a source of diagnoses):\n{ref_text}\n" if ref_text else ""
    cmgp_rule = "- If billing_model=\"PCPCM\": do NOT include CMGP time modifiers (CMGP01-10).\n"

    source_label = "TODAY'S TRANSCRIPT" if transcript_ok else "MOST RECENT DATED EMR ENTRY (FALLBACK)"
    prompt = f"""
You are an Alberta (Canada) primary care billing assistant.

TASK:
From the provided source ONLY:
1) Extract likely ICD-9 codes (max 4) strictly for diagnoses actually discussed TODAY.
2) Suggest Alberta billing codes for today's encounter. Output should be maximized but defensible.

HARD RULES:
- Do NOT include diagnoses not discussed today.
- Use ONLY the provided source text; do NOT infer from other background.
- If the source does not support a diagnosis/procedure, omit it.
- Output MUST be STRICT JSON only (no markdown, no extra text).
- If the source is an EMR fallback, treat it as the visit note for today and do NOT pull other history.
- If the source is an EMR fallback (no transcript), set icd9 = [].
{cmgp_rule}

BILLING MODEL:
- billing_model = "{model_norm}"
- If billing_model="PCPCM": treat this as SHADOW BILLING (still suggest the appropriate codes).
- If billing_model="FFS": include visit + time modifiers as appropriate, assuming {overhead_minutes} minutes extra for chart review + charting.

TIME ASSUMPTION:
- If visit length is not explicitly stated, default to the smallest reasonable CMGP modifier for a typical visit (but include the extra {overhead_minutes} minutes for FFS).
- Do NOT fabricate long visit times.

OUTPUT JSON SCHEMA:
{{
  "icd9": [{{"code":"401","dx":"Hypertension"}}],
  "billing": {{
     "codes": ["03.03A","CMGP01","93.91A"],
     "descriptors": {{"93.91A":"Hip injection"}},
     "one_line": "03.03A + CMGP01 + 93.91A (Hip injection)"
  }}
}}

NOTES:
- For "codes": include only the codes (no descriptors).
- For "one_line": include descriptors only for NON-03.03A and NON-CMGP codes, in parentheses after the code.

{source_label}:
{source_clip}
{ref_block}
""".strip()

    try:
        client = _openai_client_best_effort()
        resp = client.chat.completions.create(
            model=BILLING_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return strict JSON only. Conservative, transcript-only extraction."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"Billing OpenAI call failed: {e}")
        raw = ""

    try:
        data = json.loads(_strip_code_fences(raw))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    # ICD-9 line
    icd_parts: List[str] = []
    if transcript_ok:
        icd_parts = _extract_icd9_parts(data.get("icd9") or [])
        if not icd_parts:
            icd_parts = _extract_icd9_from_text_direct(transcript)

    line2 = f"ICD-9: {', '.join(icd_parts)}" if icd_parts else "ICD-9: "

    # Billing line
    billing = data.get("billing") or {}
    one_line = ""
    if isinstance(billing, dict):
        one_line = str(billing.get("one_line") or "").strip()

    virtual_call = bool(transcript and _is_virtual_call(transcript))
    if virtual_call:
        one_line = _normalize_billing_for_virtual(one_line, model_norm)
    elif model_norm == "PCPCM":
        one_line = _strip_cmgp_modifiers(one_line)

    line3 = f"Billing: {one_line}" if one_line else ""
    return line2, line3


def _append_entry_to_billing_text(existing_text: str, lines: List[str]) -> str:
    """
    Canonical formatting: 3 lines + blank line between entries.
    Skips empty lines to allow ICD/billing blank per spec.
    """
    lines_clean = [ln for ln in lines if (ln or "").strip()]
    if not lines_clean:
        return (existing_text or "").strip()

    next_index = _count_patients_in_billing_text(existing_text) + 1
    for i, ln in enumerate(lines_clean):
        if not ln.strip():
            continue
        if not re.match(r"^\s*\d+\.\s+", ln):
            lines_clean[i] = f"{next_index}. {ln.strip()}"
        break

    blk = "\n".join(lines_clean).strip()
    if not blk:
        return (existing_text or "").strip()
    if not (existing_text or "").strip():
        return blk
    return (existing_text.rstrip() + "\n\n" + blk).strip()


def _count_patients_in_billing_text(text: str) -> int:
    """
    Heuristic count:
    - Count entries that contain a pipe-based patient line OR a "Billing:" line.
    - Split on blank lines.
    """
    t = (text or "").strip()
    if not t:
        return 0
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", t) if b.strip()]
    n = 0
    for b in blocks:
        if "|" in b or re.search(r"(?im)^\s*Billing\s*:", b):
            n += 1
    return n


_ICD9_CODE_RE = re.compile(r"\b(\d{3}(?:\.\d{1,2})?)\b")
_BILL_CODE_RE = re.compile(
    r"\b(?:"
    r"\d{2}\.\d{2}[A-Z]{0,2}"      # 03.03A, 93.91A
    r"|03\.03CV"                  # explicit virtual visit
    r"|CMGP\d{2}"                  # CMGP01
    r"|CMXC\d{2}"                  # CMXC30
    r"|G75GP"                      # G75GP
    r"|BMIPRO"                     # BMIPRO
    r"|NBRSER"                     # NBRSER (sometimes shown as note; included to allow stripping if present)
    r")\b",
    flags=re.IGNORECASE,
)

_VIRTUAL_CALL_RE = re.compile(
    r"\b("
    r"phone call|telephone|phone visit|by phone|over the phone|on the phone"
    r"|telehealth|telemedicine|telemed|televisit|virtual visit|video visit|video call"
    r"|zoom|teams|facetime|google meet|meet\.google|whatsapp call"
    r"|phone ringing|ringtone|call connected|call ended|called patient|called pt"
    r")\b",
    flags=re.IGNORECASE,
)


def _normalize_bill_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return ""
    if re.match(r"(?i)^cmgp\d{2}$", c):
        return c.upper()
    if re.match(r"(?i)^cmxc\d{2}$", c):
        return c.upper()
    if c.lower() in ("g75gp", "bmipro", "nbrser"):
        return c.upper()
    return c


def _is_virtual_call(transcript_text: str) -> bool:
    t = (transcript_text or "").strip()
    if not t:
        return False
    low = t.lower()
    if _VIRTUAL_CALL_RE.search(low):
        return True
    if "ringing" in low and ("phone" in low or "call" in low):
        return True
    if re.match(r"^\s*(hello|hi|good (morning|afternoon|evening))\b", low):
        return True
    if re.search(r"\bhello\b", low) and re.search(r"\b(call|calling|phone|telephone|virtual|video)\b", low):
        return True
    return False


def _normalize_billing_for_virtual(one_line: str, billing_model: str) -> str:
    raw = (one_line or "").strip()
    codes = [_normalize_bill_code(c) for c in _BILL_CODE_RE.findall(raw)]
    # dedupe while preserving order
    seen = set()
    codes_norm: List[str] = []
    for c in codes:
        if not c or c in seen:
            continue
        seen.add(c)
        codes_norm.append(c)

    # Map descriptors (keep only for non-03.03/CMGP)
    desc_map: Dict[str, str] = {}
    for m in re.finditer(r"\b([A-Z0-9\.]+)\s*\(([^)]+)\)", raw):
        code = _normalize_bill_code(m.group(1))
        desc = (m.group(2) or "").strip()
        if code and desc:
            desc_map[code] = desc

    if "03.03CV" not in codes_norm:
        if "03.03A" in codes_norm:
            codes_norm = ["03.03CV" if c == "03.03A" else c for c in codes_norm]
            codes_norm = [c for c in codes_norm if c]
        else:
            codes_norm = ["03.03CV"] + codes_norm
    codes_norm = [c for c in codes_norm if c != "03.03A"]

    bm = (billing_model or "FFS").strip().upper()
    if bm == "FFS":
        if not any(c.upper().startswith("CMGP") for c in codes_norm):
            try:
                idx = codes_norm.index("03.03CV")
                codes_norm.insert(idx + 1, "CMGP01")
            except ValueError:
                codes_norm.append("CMGP01")
    else:
        codes_norm = [c for c in codes_norm if not c.upper().startswith("CMGP")]

    base = " + ".join(codes_norm).strip()
    if not base:
        return ""

    descriptors: List[str] = []
    for code, desc in desc_map.items():
        if code.startswith("CMGP") or code in ("03.03A", "03.03CV"):
            continue
        if code in codes_norm:
            descriptors.append(f"{code} ({desc})")
    if descriptors:
        base = base + "  " + "; ".join(dict.fromkeys(descriptors))
    return base.strip()


def _strip_descriptions_for_print(text: str) -> str:
    """
    Print output rules:
    - ICD-9 line: keep only codes (e.g., "ICD-9: 401, 250")
    - Billing line: keep only codes (e.g., "Billing: 03.03A + CMGP01 + 93.91A")
    """
    out_lines: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.rstrip()
        if not s.strip():
            out_lines.append("")
            continue

        if re.match(r"(?i)^\s*ICD-?9\s*:", s):
            codes = _ICD9_CODE_RE.findall(s)
            codes = [c for c in codes if c]
            if codes:
                out_lines.append("ICD-9: " + ", ".join(codes))
            else:
                out_lines.append("ICD-9:")
            continue

        if re.match(r"(?i)^\s*Billing\s*:", s):
            # Preserve order of appearance
            codes = _BILL_CODE_RE.findall(s)
            norm_codes: List[str] = []
            for c in codes:
                c2 = c.strip()
                if not c2:
                    continue
                # normalize casing: CMGP/CMXC/G75GP/BMIPRO uppercase; numeric codes keep as-is
                if re.match(r"(?i)^cmgp\d{2}$", c2):
                    c2 = c2.upper()
                elif re.match(r"(?i)^cmxc\d{2}$", c2):
                    c2 = c2.upper()
                elif c2.lower() in ("g75gp", "bmipro", "nbrser"):
                    c2 = c2.upper()
                norm_codes.append(c2)

            if norm_codes:
                out_lines.append("Billing: " + " + ".join(norm_codes))
            else:
                out_lines.append("Billing:")
            continue

        out_lines.append(s)

    # Trim trailing blank lines
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()

    return "\n".join(out_lines).strip()


# =========================
# Billing API models + endpoints
# =========================

class BillingModelPayload(BaseModel):
    billing_model: str  # "FFS" or "PCPCM"


class BillingSavePayload(BaseModel):
    billing_text: str
    physician: Optional[str] = None
    billing_model: Optional[str] = None  # allow save to also update dropdown selection


class BillingBillPayload(BaseModel):
    # optional override; if not provided, uses stored daily selection
    billing_model: Optional[str] = None


class BillingArchiveUpdatePayload(BaseModel):
    text: str


@router.get("/billing/today")
def get_billing_today(user: AuthUser = Depends(require_user)):
    """
    Returns the daily billing state for Edmonton-local "today".
    Frontend can render:
      - Physician:
      - Date:
      - Billing model:
      - Total patient count:
      - billing_text (editable area)
    """
    day_key = _today_key_edmonton()
    st = _init_daily_billing_state_if_missing(user.username, day_key)

    with BILLING_LOCK:
        billing_text = (st.get("billing_text") or "")
        physician = (st.get("physician") or "")
        billing_model = (st.get("billing_model") or "FFS").strip().upper()
        if billing_model not in ("FFS", "PCPCM"):
            billing_model = "FFS"
        total = _count_patients_in_billing_text(billing_text)

        return {
            "date": day_key,
            "physician": physician,
            "billing_model": billing_model,
            "total_patient_count": total,
            "billing_text": billing_text,
            "is_empty": (not billing_text.strip()),
            "last_updated_at": st.get("last_updated_at"),
        }


@router.post("/billing/model")
def set_billing_model_today(payload: BillingModelPayload, user: AuthUser = Depends(require_user)):
    """
    Sticky per day (Edmonton local):
    - Default: FFS
    - If set to PCPCM, keep for the rest of the day (in-memory).
    """
    day_key = _today_key_edmonton()
    st = _init_daily_billing_state_if_missing(user.username, day_key)

    model = (payload.billing_model or "").strip().upper()
    if model not in ("FFS", "PCPCM"):
        raise HTTPException(status_code=400, detail="billing_model must be 'FFS' or 'PCPCM'")

    with BILLING_LOCK:
        st["billing_model"] = model
        _touch_billing_state(user.username, st)

    return {"ok": True, "date": day_key, "billing_model": model, "last_updated_at": st.get("last_updated_at")}


@router.post("/billing/save")
def save_billing_today(payload: BillingSavePayload, user: AuthUser = Depends(require_user)):
    """
    Primary action:
    - Save the current editable billing display.
    - Optionally update physician and billing_model in the same save.
    """
    day_key = _today_key_edmonton()
    st = _init_daily_billing_state_if_missing(user.username, day_key)

    billing_text = (payload.billing_text or "").rstrip()

    with BILLING_LOCK:
        st["billing_text"] = billing_text
        if payload.physician is not None:
            st["physician"] = (payload.physician or "").strip()
        if payload.billing_model is not None:
            bm = (payload.billing_model or "").strip().upper()
            if bm in ("FFS", "PCPCM"):
                st["billing_model"] = bm
        _touch_billing_state(user.username, st)

        total = _count_patients_in_billing_text(st.get("billing_text") or "")

    return {
        "ok": True,
        "date": day_key,
        "physician": st.get("physician") or "",
        "billing_model": st.get("billing_model") or "FFS",
        "total_patient_count": total,
        "billing_text": st.get("billing_text") or "",
        "last_updated_at": st.get("last_updated_at"),
    }


@router.post("/session/{session_id}/billing/bill")
def bill_current_session_into_daily_list(
    session_id: str,
    payload: BillingBillPayload = BillingBillPayload(),
    user: AuthUser = Depends(require_user),
):
    """
    BILL action:
    - Generates a 3-line entry and auto-saves to the daily billing list (server memory).
      1) Patient line from EMR dump ONLY (Name | PHN | age in years or months if <2)
      2) ICD-9 line from transcript ONLY (blank if transcript not available)
      3) Billing line from transcript ONLY (blank if transcript not available)
    - Auto-appends and returns updated daily state.
    """
    context = _get_context_or_404(session_id)
    _ensure_anchor_hydrated_from_emr(context)

    day_key = _today_key_edmonton()
    st = _init_daily_billing_state_if_missing(user.username, day_key)

    with BILLING_LOCK:
        current_model = (payload.billing_model or st.get("billing_model") or "FFS").strip().upper()
        if current_model not in ("FFS", "PCPCM"):
            current_model = "FFS"
        st["billing_model"] = current_model

    # Line 1 from EMR
    line1, _ids = _patient_line_from_emr(context)

    # Lines 2-3 from transcript (strict)
    line2, line3 = _generate_icd9_and_billing_lines(context, current_model)

    lines = [line1, line2, line3]

    with BILLING_LOCK:
        st["billing_text"] = _append_entry_to_billing_text(st.get("billing_text") or "", lines)
        _touch_billing_state(user.username, st)
        total = _count_patients_in_billing_text(st.get("billing_text") or "")

        return {
            "ok": True,
            "date": day_key,
            "physician": st.get("physician") or "",
            "billing_model": st.get("billing_model") or "FFS",
            "total_patient_count": total,
            "billing_text": st.get("billing_text") or "",
            "last_updated_at": st.get("last_updated_at"),
            "appended_entry": "\n".join([ln for ln in lines if (ln or "").strip()]).strip(),
        }


@router.post("/billing/print")
def print_and_clear_billing_today(user: AuthUser = Depends(require_user)):
    """
    PRINT action:
    - Returns a print-ready text version of the billing list that strips:
        - ICD-9 descriptions (keeps codes only)
        - Billing code descriptions (keeps codes only)
    - After generating print output, clears the stored billing list for the day.
      (Billing model selection remains for the day.)
    """
    day_key = _today_key_edmonton()
    st = _init_daily_billing_state_if_missing(user.username, day_key)

    archive_name = ""
    saved_at_local = ""
    saved_at_utc = ""

    with BILLING_LOCK:
        raw_text = (st.get("billing_text") or "").strip()
        printable = _strip_descriptions_for_print(raw_text)
        total = _count_patients_in_billing_text(raw_text)

        # Clear only the billing memory (list/display); keep physician + model sticky for day
        st["billing_text"] = ""
        _touch_billing_state(user.username, st)

    if printable:
        dt_local = _now_edmonton()
        saved_at_local = dt_local.isoformat()
        saved_at_utc = _utcnow().isoformat()
        archive_dir = _billing_archive_dir(user.username)
        archive_name = _make_billing_archive_filename(archive_dir, dt_local)
        archive_path = os.path.join(archive_dir, archive_name)
        try:
            with open(archive_path, "w", encoding="utf-8") as f:
                f.write(printable)
        except Exception as e:
            logger.warning(f"Failed to save billing archive for {user.username}: {e}")

    return {
        "ok": True,
        "date": day_key,
        "billing_model": (st.get("billing_model") or "FFS"),
        "physician": (st.get("physician") or ""),
        "print_text": printable,
        "total_patient_count": total,
        "cleared": True,
        "archive_filename": archive_name,
        "archive_saved_at_local": saved_at_local,
        "archive_saved_at_utc": saved_at_utc,
        "last_updated_at": st.get("last_updated_at"),
    }


@router.get("/billing/archives")
def list_billing_archives(user: AuthUser = Depends(require_user)):
    base = _billing_archive_dir(user.username)
    items: List[Dict[str, Any]] = []
    try:
        names = [n for n in os.listdir(base) if n.endswith(".txt")]
    except FileNotFoundError:
        names = []
    for name in names:
        path = os.path.join(base, name)
        try:
            st = os.stat(path)
        except Exception:
            continue
        dt_local = datetime.fromtimestamp(st.st_mtime, EDMONTON_TZ)
        dt_utc = datetime.fromtimestamp(st.st_mtime, timezone.utc)
        items.append({
            "filename": name,
            "size_bytes": st.st_size,
            "saved_at_local": dt_local.isoformat(),
            "saved_at_utc": dt_utc.isoformat(),
        })
    items.sort(key=lambda x: x.get("saved_at_local", ""), reverse=True)
    return {"items": items}


@router.get("/billing/archives/{filename}")
def get_billing_archive(filename: str, user: AuthUser = Depends(require_user)):
    path = _safe_billing_archive_path(user.username, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Billing archive not found.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read billing archive.")
    total = _count_patients_in_billing_text(text)
    st = os.stat(path)
    dt_local = datetime.fromtimestamp(st.st_mtime, EDMONTON_TZ)
    dt_utc = datetime.fromtimestamp(st.st_mtime, timezone.utc)
    return {
        "filename": filename,
        "text": text,
        "total_patient_count": total,
        "saved_at_local": dt_local.isoformat(),
        "saved_at_utc": dt_utc.isoformat(),
    }


@router.put("/billing/archives/{filename}")
def update_billing_archive(filename: str, payload: BillingArchiveUpdatePayload, user: AuthUser = Depends(require_user)):
    path = _safe_billing_archive_path(user.username, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Billing archive not found.")
    text = (payload.text or "").rstrip()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to update billing archive.")
    total = _count_patients_in_billing_text(text)
    st = os.stat(path)
    dt_local = datetime.fromtimestamp(st.st_mtime, EDMONTON_TZ)
    dt_utc = datetime.fromtimestamp(st.st_mtime, timezone.utc)
    return {
        "ok": True,
        "filename": filename,
        "total_patient_count": total,
        "saved_at_local": dt_local.isoformat(),
        "saved_at_utc": dt_utc.isoformat(),
    }


@router.delete("/billing/archives/{filename}")
def delete_billing_archive(filename: str, user: AuthUser = Depends(require_user)):
    path = _safe_billing_archive_path(user.username, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Billing archive not found.")
    try:
        os.remove(path)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete billing archive.")
    return {"deleted": True, "filename": filename}
