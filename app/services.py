from __future__ import annotations

from datetime import datetime, timezone, date
import hashlib
import inspect
import json
import logging
import os
import re
import uuid
import threading
import html as html_lib
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List, Generator
from zoneinfo import ZoneInfo

from openai import OpenAI
from app.models import SessionContext
from app.local_kb import KB_ENABLED, search_kb, format_kb_context
from app.guideline_runner import run_guideline_runner


# =========================
# Logging
# =========================
logger = logging.getLogger("centaurweb.services")


# =========================
# OpenAI client
# =========================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# Model selection
# =========================
CLINICAL_QUERY_TEXT_MODEL = "gpt-5-nano"
CLINICAL_QUERY_FAST_MODEL = os.getenv("CLINICAL_QUERY_FAST_MODEL", "gpt-5-nano")
CLINICAL_QUERY_THINK_MODEL = os.getenv("CLINICAL_QUERY_THINK_MODEL", CLINICAL_QUERY_TEXT_MODEL)
CLINICAL_QUERY_VISION_MODEL = os.getenv("CLINICAL_QUERY_VISION_MODEL", "gpt-5-nano")
CLINICAL_QUERY_VISION_FAST_MODEL = os.getenv("CLINICAL_QUERY_VISION_FAST_MODEL", "gpt-5-nano")

# Patient summary model
PATIENT_SUMMARY_MODEL = os.getenv("PATIENT_SUMMARY_MODEL", "gpt-5-nano")
PATIENT_SUMMARY_FALLBACK_MODEL = os.getenv("PATIENT_SUMMARY_FALLBACK_MODEL", "gpt-5.2")
PATIENT_SUMMARY_MAX_CHARS = int(os.getenv("PATIENT_SUMMARY_MAX_CHARS", "16000"))

# Billing model (use a smaller model; upgrade if you prefer)
BILLING_MODEL = os.getenv("BILLING_MODEL", "gpt-5.2")

# Web search (guidelines)
WEB_SEARCH_ENABLED = (os.getenv("CENTAUR_WEB_SEARCH", "1").strip() == "1")
WEB_SEARCH_TIMEOUT_SEC = float(os.getenv("CENTAUR_WEB_SEARCH_TIMEOUT_SEC", "7"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("CENTAUR_WEB_SEARCH_MAX_RESULTS", "4"))
WEB_SEARCH_USER_AGENT = os.getenv(
    "CENTAUR_WEB_SEARCH_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
)

# Clinical Query speed controls
CLINICAL_QUERY_SERVICE_TIER = os.getenv("CENTAUR_CQ_SERVICE_TIER", "priority").strip() or "priority"
CLINICAL_QUERY_MAX_TOKENS = int(os.getenv("CENTAUR_CQ_MAX_TOKENS", "450"))
CLINICAL_QUERY_USE_WEB = (os.getenv("CENTAUR_CQ_WEB", "0").strip() == "1")


# =========================
# Time helpers (timezone-safe)
# =========================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _months_between(dob: date, today: date) -> int:
    """
    Whole-month difference between dob and today.
    """
    months = (today.year - dob.year) * 12 + (today.month - dob.month)
    if today.day < dob.day:
        months -= 1
    return max(0, months)


# =========================
# Utilities
# =========================

def _hash_context(context: SessionContext) -> str:
    """
    Stable hash for caching/debug. Works with Pydantic v1/v2.
    """
    try:
        payload_obj = context.model_dump()  # Pydantic v2
    except Exception:
        payload_obj = context.dict()  # Pydantic v1 / compat
    payload = json.dumps(payload_obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_patient_obj(context: SessionContext):
    """
    Compatibility helper:
    - Prefer context.patient_anchor (canonical identifiers)
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


def _try_init_patient_anchor(context: SessionContext) -> None:
    """
    Best-effort initializer so identifier hydration doesn't silently no-op
    when patient_anchor is missing/None. Safe if models differ across versions.
    """
    try:
        if not hasattr(context, "patient_anchor"):
            return
        if getattr(context, "patient_anchor", None) is not None:
            return

        # Pydantic v2: model_fields
        try:
            mf = getattr(context.__class__, "model_fields", None)
            if isinstance(mf, dict) and "patient_anchor" in mf:
                anno = getattr(mf["patient_anchor"], "annotation", None)
                if anno is not None:
                    try:
                        setattr(context, "patient_anchor", anno())  # type: ignore[misc]
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # Pydantic v1: __fields__
        try:
            f1 = getattr(context.__class__, "__fields__", None)
            if isinstance(f1, dict) and "patient_anchor" in f1:
                anno = getattr(f1["patient_anchor"], "type_", None)
                if anno is not None:
                    try:
                        setattr(context, "patient_anchor", anno())  # type: ignore[misc]
                        return
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception:
        return


def _get_str_attr(obj, attr: str) -> str:
    if obj is None:
        return ""
    v = getattr(obj, attr, "") or ""
    return str(v).strip()


def _field_exists(obj, attr: str) -> bool:
    """
    True if attr is a declared field on a Pydantic model (v1 or v2), else best-effort.
    """
    if obj is None:
        return False
    try:
        mf = getattr(obj, "model_fields", None)  # pydantic v2
        if isinstance(mf, dict):
            return attr in mf
    except Exception:
        pass
    try:
        f1 = getattr(obj, "__fields__", None)  # pydantic v1
        if isinstance(f1, dict):
            return attr in f1
    except Exception:
        pass

    # Fallback: if it already exists as an attribute, allow
    try:
        return hasattr(obj, attr)
    except Exception:
        return False


def _set_attr(obj, attr: str, value, overwrite: bool = False) -> bool:
    """
    Safe setter with schema checks.

    overwrite=False:
      - only sets if current is None or "" (string)
    overwrite=True:
      - sets if value is non-empty and different from current (or current is empty)

    Returns True if updated.
    """
    if obj is None:
        return False
    if not _field_exists(obj, attr):
        return False

    try:
        cur = getattr(obj, attr, None)
    except Exception:
        cur = None

    # Decide "empty"
    cur_empty = cur is None or (isinstance(cur, str) and not cur.strip())

    if not overwrite:
        if cur_empty:
            try:
                setattr(obj, attr, value)
                return True
            except Exception:
                return False
        return False

    # overwrite=True
    # Do not overwrite with empty value
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False

    # Only write if different OR current empty
    different = True
    try:
        if isinstance(cur, str) and isinstance(value, str):
            different = cur.strip() != value.strip()
        else:
            different = cur != value
    except Exception:
        different = True

    if cur_empty or different:
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False

    return False


def _clip_text(text: str, max_chars: int = 4000) -> str:
    """
    Clip long text while preserving both head and tail (tail often contains most recent info).
    """
    t = (text or "").strip()
    if not t:
        return ""
    if max_chars <= 0 or len(t) <= max_chars:
        return t

    head = max_chars // 2
    tail = max_chars - head
    return t[:head].rstrip() + "\n\n[...clipped...]\n\n" + t[-tail:].lstrip()


# =========================
# Web search helpers
# =========================

def _sanitize_web_query(text: str, max_len: int = 180) -> str:
    """
    Remove obvious identifiers (PHN, long numbers, names) from search queries.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Drop long digit runs (PHN, phone, IDs)
    t = re.sub(r"\b\d{3,}\b", " ", t)
    # Drop common name pattern: First Last
    t = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", " ", t)
    # Keep only safe chars
    t = re.sub(r"[^A-Za-z0-9\s\-\.\,\/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[:max_len].rstrip()
    return t


def _web_search(query: str) -> List[Dict[str, str]]:
    if not WEB_SEARCH_ENABLED:
        return []
    q = _sanitize_web_query(query)
    if not q:
        return []

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
    req = urllib.request.Request(url, headers={"User-Agent": WEB_SEARCH_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=WEB_SEARCH_TIMEOUT_SEC) as resp:
            raw = resp.read(500_000)
    except Exception:
        return []

    html_text = raw.decode("utf-8", errors="ignore")

    results: List[Dict[str, str]] = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text):
        href = html_lib.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2) or "")
        title = html_lib.unescape(title).strip()

        snippet = ""
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
            html_text[m.end(): m.end() + 900],
            flags=re.S,
        )
        if snippet_match:
            snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1) or "")
            snippet = html_lib.unescape(snippet).strip()

        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= WEB_SEARCH_MAX_RESULTS:
            break

    return results


def _format_web_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "None"
    lines: List[str] = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if snippet:
            lines.append(f"{i}. {title} — {url}\nSnippet: {snippet}")
        else:
            lines.append(f"{i}. {title} — {url}")
    return "\n".join(lines).strip()


def _build_web_query_for_clinical_query(context: SessionContext, query: str) -> str:
    base = "clinical guideline Canada Alberta primary care"
    age = ""
    sex = ""
    try:
        pa = getattr(context, "patient_anchor", None)
        if pa:
            if getattr(pa, "age", None) is not None:
                age = str(getattr(pa, "age"))
            if getattr(pa, "sex", None):
                sex = str(getattr(pa, "sex")).lower()
    except Exception:
        pass
    demo = " ".join([x for x in [age, sex] if x])
    core = _sanitize_web_query(query)
    if demo:
        return f"{core} {demo} {base}".strip()
    return f"{core} {base}".strip()


def _build_web_query_for_coach(context: SessionContext) -> str:
    base = "latest guideline Canada Alberta primary care"
    seed = ""
    try:
        tx = (getattr(context.transcript, "raw_text", None) or "").strip()
        bg = (getattr(context.clinical_background, "emr_dump", None) or "").strip()
        seed = tx if len(tx) >= 80 else bg
    except Exception:
        seed = ""
    seed = _sanitize_web_query(seed)
    if seed:
        return f"{seed} {base}".strip()
    return base


def _safe_getattr_chain(obj, path: str, default=None):
    """
    Safely access nested attributes: "derived_outputs.patient_summary"
    """
    cur = obj
    for part in (path or "").split("."):
        if not part:
            continue
        if cur is None:
            return default
        try:
            cur = getattr(cur, part)
        except Exception:
            return default
    return cur if cur is not None else default


def _fn_accepts_kw(fn, name: str) -> bool:
    """
    Returns True if fn accepts keyword 'name' or **kwargs.
    Used for backwards-compatible optional parameters.
    """
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


def _normalize_mime(m: str) -> str:
    return (m or "").lower().strip()


def _is_image_mime(m: str) -> bool:
    m = _normalize_mime(m)
    return m.startswith("image/")


def _attachments_image_summary(attachments: Optional[List[Dict[str, Any]]]) -> str:
    """
    Human-readable summary for the prompt (filenames only).
    """
    if not attachments:
        return "None"
    names: List[str] = []
    for a in attachments:
        if not isinstance(a, dict):
            continue
        if (a.get("kind") or "").lower().strip() != "image":
            continue
        fn = (a.get("filename") or a.get("id") or "").strip()
        if fn:
            names.append(fn)
    if not names:
        return "None"
    if len(names) > 8:
        names = names[:8] + ["..."]
    return ", ".join(names)


def _build_multimodal_user_content(prompt_text: str, attachments: Optional[List[Dict[str, Any]]]) -> Any:
    """
    Returns content suitable for OpenAI Chat Completions.
    If images exist, returns a list of content parts: text + image_url parts.
    Otherwise returns a plain string.
    """
    if not attachments:
        return prompt_text

    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    for a in attachments:
        if not isinstance(a, dict):
            continue
        if (a.get("kind") or "").lower().strip() != "image":
            continue
        mime = _normalize_mime(a.get("mime") or "") or "image/jpeg"
        if not _is_image_mime(mime):
            continue
        b64 = (a.get("data_base64") or "").strip()
        if not b64:
            continue

        url = f"data:{mime};base64,{b64}"
        parts.append({"type": "image_url", "image_url": {"url": url}})

    if len(parts) == 1:
        return prompt_text

    return parts


# =========================
# DEMOGRAPHICS EXTRACTION
# =========================
# IMPORTANT: EMR formats vary widely; patterns must be tolerant.

NAME_PATTERNS = [
    r"(?:^|\n)\s*(?:Patient\s*Name|Patient|Name|Pt)\s*[:\-]\s*([^\n]{3,80})",
    r"(?:^|\n)\s*([A-Z]{2,})\s*,\s*([A-Z]{2,}|[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3})\b",
]

# Alberta PHN is 9 digits. Ava-style example: "396906602 | AB"
PHN_AB_PIPE_PATTERNS = [
    r"\b([0-9]{9})\s*\|\s*AB\b",
    r"\bAB\s*\|\s*([0-9]{9})\b",
]

PHN_LABEL_PATTERNS = [
    r"\bPHN\b\s*[:#]?\s*([0-9][0-9\-\s]{7,20})\b",
    r"\bPersonal\s+Health\s+Number\b\s*[:#]?\s*([0-9][0-9\-\s]{7,20})\b",
    r"\bHealth\s*(?:Care|Card)?\s*Number\b\s*[:#]?\s*([0-9][0-9\-\s]{7,20})\b",
    r"\bPHN\s*/\s*ULI\b\s*[:#]?\s*([0-9][0-9\-\s]{7,20})\b",
    r"\bULI\b\s*[:#]?\s*([0-9][0-9\-\s]{7,20})\b",
]

DOB_PATTERNS = [
    r"\bDOB\b\s*[:#]?\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})\b",
    r"\bDate\s+of\s+Birth\b\s*[:#]?\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2})\b",
    r"\bDOB\b\s*[:#]?\s*([0-9]{4}[-/][A-Za-z]{3,9}[-/][0-9]{1,2})\b",
    r"\bDate\s+of\s+Birth\b\s*[:#]?\s*([0-9]{4}[-/][A-Za-z]{3,9}[-/][0-9]{1,2})\b",
    r"\bDOB\b\s*[:#]?\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b",
    r"\bDate\s+of\s+Birth\b\s*[:#]?\s*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b",
    r"\bDOB\b\s*[:#]?\s*([0-9]{1,2}[-/][A-Za-z]{3,9}[-/][0-9]{2,4})\b",
    r"\bDate\s+of\s+Birth\b\s*[:#]?\s*([0-9]{1,2}[-/][A-Za-z]{3,9}[-/][0-9]{2,4})\b",
    r"\bDOB\b\s*[:#]?\s*([A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4})\b",
    r"\bDate\s+of\s+Birth\b\s*[:#]?\s*([A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4})\b",
    r"\bDOB\b\s*[:#]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})\b",
    r"\bDate\s+of\s+Birth\b\s*[:#]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})\b",
]

AGE_PATTERNS = [
    r"\bAge\b\s*[:#]?\s*([0-9]{1,3})\b",
    r"\b([0-9]{1,3})\s*(?:yr|yrs|year|years)\s*old\b",
    r"\b([0-9]{1,3})\s*(?:yo|y/o)\b",
]

SEX_PATTERNS = [
    r"\bSex\b\s*[:#]?\s*(male|female|other|unknown|m|f)\b",
    r"\bGender\b\s*[:#]?\s*(male|female|other|unknown|m|f)\b",
    r"\b(?:[0-9]{1,3}\s*(?:yr|yrs|year|years)\s*old)\s*(male|female)\b",
    r"\b(?:[0-9]{1,3}\s*(?:yo|y/o))\s*(male|female)\b",
    r"\b(?:[0-9]{1,3}[- ]?year[- ]?old)\s*(male|female)\b",
]

# Phone numbers: exclude from PHN candidate selection
_PHONE_RE = re.compile(
    r"(?:(?:\+?1[\s\-\.])?)"
    r"(?:\(?\b\d{3}\)?[\s\-\.]?)"
    r"\b\d{3}[\s\-\.]?\d{4}\b"
)

# Provider contamination control (prevents "Thapa, Rajat" etc. from being treated as patient name)
_PROVIDER_STOPWORDS = {
    "dr", "doctor", "author",
    "one", "health", "associate", "medical", "inc", "clinic",
    "thapa", "rajat",
}
_PROVIDER_LINE_KEYWORDS = [
    "author:",
    "dr.",
    "dr ",
    "one health",
    "associate medical",
    "inc.",
    "inc)",
    "sincerely",
]
_PROVINCE_ABBR = {
    "ab", "bc", "mb", "nb", "nl", "ns", "nt", "nu", "on", "pe", "qc", "sk", "yt",
}


def _clean_phn(phn: str) -> str:
    """
    Clean a PHN candidate into Alberta-friendly digits.

    - Prefer exactly 9 digits.
    - Accept leading zero padding (e.g., 00 + 9 digits) by stripping leading zeros to 9 digits.
    - Avoid "take last 9 digits" behavior for arbitrary long numbers (prevents phone-number truncation errors).
    """
    raw = (phn or "").strip()
    digits = re.sub(r"\D+", "", raw)

    if len(digits) == 9:
        return digits

    # If padded with leading zeros, allow: e.g. '00' + 9 digits
    if len(digits) > 9:
        # allow only if the prefix is all zeros
        prefix = digits[:-9]
        if prefix and set(prefix) <= {"0"}:
            return digits[-9:]
        # otherwise reject (too risky)
        return ""

    # If under-length, keep digits (best-effort), but will be treated as unreliable downstream
    if len(digits) >= 7:
        return digits

    return ""


def _phone_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in _PHONE_RE.finditer(text or ""):
        spans.append((m.start(), m.end()))
    return spans


def _overlaps(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
    s0, e0 = span
    for s1, e1 in spans:
        if s0 < e1 and e0 > s1:
            return True
    return False


def _extract_phn_best_effort(text: str) -> str:
    """
    Alberta-focused PHN extraction.

    Priority order:
      1) Ava-style: "<9 digits> | AB"
      2) Labelled PHN/ULI patterns (PHN:, Personal Health Number, etc.)
      3) Scored fallback: standalone 9-digit candidates that are NOT phone numbers,
         preferring proximity to AB / AHC / PHN tokens.
    """
    t = text or ""
    if not t.strip():
        return ""

    # 1) Ava-style: digits | AB
    for p in PHN_AB_PIPE_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1):
            c = _clean_phn(m.group(1))
            if len(c) == 9:
                return c

    # 2) Labelled patterns
    for p in PHN_LABEL_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1):
            c = _clean_phn(m.group(1))
            if len(c) == 9:
                return c

    # 3) Scored fallback among 9-digit sequences, excluding phone spans
    phone_sp = _phone_spans(t)

    candidates: List[Tuple[int, str, int]] = []  # (pos, value, score)
    for m in re.finditer(r"\b([0-9]{9})\b", t):
        span = (m.start(1), m.end(1))
        if _overlaps(span, phone_sp):
            continue
        val = m.group(1)
        if not val:
            continue

        # Context window scoring
        w0 = max(0, span[0] - 40)
        w1 = min(len(t), span[1] + 40)
        ctx = t[w0:w1].lower()

        score = 0
        if "| ab" in ctx or "ab |" in ctx:
            score += 12
        if " phn" in ctx or "phn" in ctx:
            score += 8
        if "personal health" in ctx:
            score += 6
        if "ahc" in ctx or "health care" in ctx:
            score += 5

        # Penalize if looks like phone context
        if "tel" in ctx or "phone" in ctx or "fax" in ctx:
            score -= 10

        candidates.append((span[0], val, score))

    if not candidates:
        return ""

    # pick highest score; tie-breaker: earliest occurrence
    candidates.sort(key=lambda x: (-x[2], x[0]))
    best = candidates[0]
    if best[2] <= 0:
        return ""
    return best[1]


def _title_case_name(s: str) -> str:
    return " ".join([w.capitalize() for w in re.split(r"\s+", (s or "").strip()) if w])


def _collapse_duplicate_tokens(name: str) -> str:
    toks = [t for t in re.split(r"\s+", (name or "").strip()) if t]
    out: List[str] = []
    for t in toks:
        if not out or out[-1].lower() != t.lower():
            out.append(t)
    return " ".join(out)


def _looks_like_provider_line(line: str) -> bool:
    low = (line or "").strip().lower()
    if not low:
        return False
    return any(k in low for k in _PROVIDER_LINE_KEYWORDS)


def _name_has_provider_stopwords(name: str) -> bool:
    toks = [t.lower() for t in re.split(r"\s+", (name or "").strip()) if t]
    return any(t in _PROVIDER_STOPWORDS for t in toks)


def _sanitize_patient_name(name: str) -> str:
    """
    Final guard: prevent clinician/clinic contamination. Return "" if unsafe.
    """
    n = (name or "").strip()
    if not n:
        return ""
    if _name_has_provider_stopwords(n):
        return ""
    toks = [t.lower() for t in re.split(r"\s+", n) if t]
    if any(t in _PROVINCE_ABBR for t in toks):
        return ""
    low = n.lower()
    if "one health" in low or "associate medical" in low:
        return ""
    return _collapse_duplicate_tokens(n).strip()


def _extract_patient_name_near_phn(text: str, phn: str) -> str:
    """
    Deterministic extraction:
      - If PHN exists, find its line and scan upward for the nearest "LAST, First".
      - Exclude provider/clinic lines and stopword contamination.
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    phn = _clean_phn(phn or "")
    if not t.strip() or len(phn) != 9:
        return ""

    m_phn = re.search(rf"\b{re.escape(phn)}\b", t)
    if not m_phn:
        return ""

    # Work in a bounded window around PHN; avoids note bodies where author names appear frequently.
    start = max(0, m_phn.start() - 3000)
    end = min(len(t), m_phn.end() + 300)
    region = t[start:end]

    lines = [ln.strip() for ln in region.split("\n") if ln.strip()]
    if not lines:
        return ""

    # Find line index containing PHN
    idx_phn = 0
    for i, ln in enumerate(lines):
        if phn in re.sub(r"\D+", "", ln):
            idx_phn = i
            break

    # Scan upwards from PHN line
    name_re = re.compile(
        r"^\s*([A-Z]{2,40}|[A-Z][A-Za-z'\-]{1,40})\s*,\s*([A-Z][A-Za-z'\-]{1,40})"
        r"(?:\s+\([A-Za-z'\-]{1,40}\))?(?:\s+[A-Z][A-Za-z'\-]{1,40}){0,2}\b"
    )
    for i in range(idx_phn, max(-1, idx_phn - 40), -1):
        ln = lines[i]
        if _looks_like_provider_line(ln):
            continue
        m = name_re.match(ln)
        if not m:
            continue
        last_raw = (m.group(1) or "").strip()
        first_raw = (m.group(2) or "").strip()
        if not first_raw or not last_raw:
            continue

        candidate = _title_case_name(f"{first_raw} {last_raw}")
        candidate = _sanitize_patient_name(candidate)
        if candidate:
            return candidate

    return ""


def _clean_name(s: str) -> str:
    """
    Remove trailing demographics fragments that often appear on the same line.
    """
    t = re.sub(r"\s+", " ", (s or "").strip())
    t = re.sub(r"\s+(PHN|DOB|Sex|Gender)\s*[:\-].*$", "", t, flags=re.IGNORECASE).strip()
    return t


def _extract_patient_name_fallback_banner(text: str) -> str:
    """
    Fallback when PHN missing: look only near the top "banner-ish" area,
    and apply the same contamination guards.
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not t.strip():
        return ""

    banner = "\n".join(t.split("\n")[:140])
    lines = [ln.strip() for ln in banner.split("\n") if ln.strip()]
    if not lines:
        return ""

    name_re = re.compile(
        r"^\s*([A-Z]{2,40}|[A-Z][A-Za-z'\-]{1,40})\s*,\s*([A-Z][A-Za-z'\-]{1,40})"
        r"(?:\s+\([A-Za-z'\-]{1,40}\))?(?:\s+[A-Z][A-Za-z'\-]{1,40}){0,2}\b"
    )
    for ln in lines:
        if _looks_like_provider_line(ln):
            continue
        m = name_re.match(ln)
        if m:
            last_raw = (m.group(1) or "").strip()
            first_raw = (m.group(2) or "").strip()
            candidate = _title_case_name(f"{first_raw} {last_raw}")
            candidate = _sanitize_patient_name(candidate)
            if candidate:
                return candidate

        # Allow "Patient Name: X" style as last resort (still sanitized)
        m2 = re.search(NAME_PATTERNS[0], "\n" + ln, flags=re.IGNORECASE | re.MULTILINE)
        if m2 and m2.group(1):
            candidate = _sanitize_patient_name(_clean_name(m2.group(1)))
            if candidate:
                return candidate

    return ""


def _parse_date_any(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None

    fmts = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%b-%d",
        "%Y-%B-%d",
        "%Y/%b/%d",
        "%Y/%B/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%d/%b/%Y",
        "%d/%B/%Y",
        "%b %d %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
    )

    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue

    s2 = re.sub(r"\s+", " ", s)
    for fmt in ("%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s2, fmt).date()
        except Exception:
            continue

    return None


def _extract_age_best_effort(text: str, phn: str = "") -> Optional[int]:
    """
    Prefer age near the PHN/banner area to avoid picking older letter ages.
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not t.strip():
        return None

    # Try within a window around PHN line first
    phn_clean = _clean_phn(phn or "")
    if phn_clean:
        m_phn = re.search(rf"\b{re.escape(phn_clean)}\b", t)
        if m_phn:
            start = max(0, m_phn.start() - 1200)
            end = min(len(t), m_phn.end() + 1200)
            region = t[start:end]
            for p in AGE_PATTERNS:
                m = re.search(p, region, flags=re.IGNORECASE | re.MULTILINE)
                if m and m.group(1):
                    try:
                        return int(m.group(1))
                    except Exception:
                        pass

    # Fallback: scan only the banner/top area
    banner = "\n".join(t.split("\n")[:200])
    for p in AGE_PATTERNS:
        m = re.search(p, banner, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1):
            try:
                return int(m.group(1))
            except Exception:
                pass

    return None


def _age_from_dob(dob: date) -> Optional[int]:
    try:
        today = date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        if age < 0:
            return None
        return age
    except Exception:
        return None


def _normalize_sex(s: str) -> str:
    t = (s or "").strip().lower()
    if t in ("m", "male", "man", "boy"):
        return "male"
    if t in ("f", "female", "woman", "girl"):
        return "female"
    if t in ("other", "unknown"):
        return t
    return ""


def _gender_display(sex: str) -> str:
    t = (sex or "").strip().lower()
    if t == "male":
        return "Male"
    if t == "female":
        return "Female"
    if t == "other":
        return "Non-binary"
    return ""


def _age_display_from_demographics(dob_str: str, age_years: Optional[int]) -> str:
    """
    Returns "X years" or "X months" (if < 24 months and DOB known).
    """
    if dob_str:
        dob = _parse_date_any(dob_str)
        if dob:
            months = _months_between(dob, date.today())
            if months < 24:
                return f"{months} months" if months != 1 else "1 month"
            years = _age_from_dob(dob)
            if years is not None:
                return f"{years} years" if years != 1 else "1 year"

    if age_years is None:
        return ""
    if age_years < 2 and dob_str:
        return f"{age_years} years" if age_years != 1 else "1 year"
    return f"{age_years} years" if age_years != 1 else "1 year"


def extract_demographics_from_text(text: str) -> Tuple[str, str, str, Optional[int], Optional[str]]:
    """
    Returns (name, phn, dob, age, sex). Any field may be empty/None if not found.

    IMPORTANT BEHAVIOR CHANGE:
    - PHN is extracted first.
    - Patient name is then extracted deterministically near PHN (or banner fallback),
      with explicit filtering to prevent clinician name contamination (e.g., "Thapa, Rajat").
    """
    t = text or ""

    # Alberta PHN extraction (supports: "396906602 | AB")
    phn = _extract_phn_best_effort(t)

    # Deterministic patient name extraction to prevent provider name contamination
    name = ""
    if phn:
        name = _extract_patient_name_near_phn(t, phn)
    if not name:
        name = _extract_patient_name_fallback_banner(t)

    # Final fallback: legacy patterns, but sanitized (last resort only)
    if not name:
        for p in NAME_PATTERNS:
            m = re.search(p, t, flags=re.IGNORECASE | re.MULTILINE)
            if not m:
                continue
            if len(m.groups()) == 1:
                candidate = _sanitize_patient_name(_clean_name(m.group(1) or ""))
                if candidate:
                    name = candidate
                    break
            elif len(m.groups()) >= 2:
                last = (m.group(1) or "").strip()
                first = (m.group(2) or "").strip()
                if first and last:
                    candidate = _sanitize_patient_name(_clean_name(f"{first} {last}"))
                    if candidate:
                        name = candidate
                        break

    dob = ""
    for p in DOB_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1):
            dob = (m.group(1) or "").strip()
            break

    age: Optional[int] = _extract_age_best_effort(t, phn)

    if age is None and dob:
        dob_dt = _parse_date_any(dob)
        if dob_dt:
            age = _age_from_dob(dob_dt)

    sex: Optional[str] = None
    for p in SEX_PATTERNS:
        m = re.search(p, t, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.group(1):
            s = _normalize_sex(m.group(1))
            sex = s if s else None
            break

    return name, phn, dob, age, sex


def update_demographics_from_text(context: SessionContext, text: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Hydrate PatientAnchor using text.

    overwrite=False (default):
      - Fill only missing fields (legacy behavior).

    overwrite=True:
      - EMR-as-source-of-truth behavior: if EMR has identifiers, they overwrite
        existing non-empty anchor values.

    Returns extracted identifiers for debugging/logging.
    """
    t = (text or "").strip()
    if not t:
        return {"name": "", "phn": "", "dob": "", "age": None, "sex": ""}

    name, phn, dob, age, sex = extract_demographics_from_text(t)

    _try_init_patient_anchor(context)
    patient = _get_patient_obj(context)

    changed = False
    if patient is not None:
        if name:
            changed = _set_attr(patient, "name", name, overwrite=overwrite) or changed
        if phn:
            changed = _set_attr(patient, "phn", phn, overwrite=overwrite) or changed
        if dob:
            changed = _set_attr(patient, "dob", dob, overwrite=overwrite) or changed
        if age is not None:
            changed = _set_attr(patient, "age", age, overwrite=overwrite) or changed

        if sex:
            # Support either `sex` or `gender` field name in PatientAnchor.
            if _set_attr(patient, "sex", sex, overwrite=overwrite):
                changed = True
            elif _set_attr(patient, "gender", sex, overwrite=overwrite):
                changed = True

    if changed:
        try:
            context.session_meta.last_updated_at = _now_utc()
        except Exception:
            pass

    return {"name": name or "", "phn": phn or "", "dob": dob or "", "age": age, "sex": sex or ""}


def _extract_identifiers_from_background(context: SessionContext) -> Dict[str, Any]:
    """
    Single source of truth for identifiers in api.py and billing.

    Behavior:
    - Extract identifiers ONLY from EMR/background.
    - Write-back to patient_anchor/patient using overwrite=True because EMR is the source of truth.
      This prevents stale identifiers persisting when sessions are reused across patients.
    - Also returns extracted identifiers so billing can still return PHN
      even if patient_anchor is missing for some reason.
    """
    bg = ""
    try:
        bg = (context.clinical_background.emr_dump or "").strip()
    except Exception:
        bg = ""

    name, phn, dob, age, sex = extract_demographics_from_text(bg)

    _try_init_patient_anchor(context)
    patient = _get_patient_obj(context)

    changed = False
    if patient is not None:
        if name:
            changed = _set_attr(patient, "name", name, overwrite=True) or changed
        if phn:
            changed = _set_attr(patient, "phn", phn, overwrite=True) or changed
        if dob:
            changed = _set_attr(patient, "dob", dob, overwrite=True) or changed
        if age is not None:
            changed = _set_attr(patient, "age", age, overwrite=True) or changed
        if sex:
            if _set_attr(patient, "sex", sex, overwrite=True):
                changed = True
            elif _set_attr(patient, "gender", sex, overwrite=True):
                changed = True

    if changed:
        try:
            context.session_meta.last_updated_at = _now_utc()
        except Exception:
            pass

    return {
        "name": name or "",
        "phn": phn or "",
        "dob": dob or "",
        "age": age,
        "sex": sex or "",
    }


def _hydrate_identifiers_from_background_only(context: SessionContext) -> None:
    _extract_identifiers_from_background(context)


# Back-compat name used throughout the file
def _hydrate_identifiers_best_effort(context: SessionContext) -> None:
    """
    Deterministic backstop used by multiple endpoints.

    Behavior:
    - ONLY uses EMR/background for identifiers (Name/PHN/DOB/Age/Sex).
    - Does NOT parse identifiers from transcript.
    """
    _hydrate_identifiers_from_background_only(context)


# =========================
# FACT PROMOTION
# =========================

FACT_KEYWORDS = [
    "med", "meds", "medication", "dose", "mg", "bid", "tid", "daily",
    "pmhx", "hx", "history", "diagnosis", "dx",
    "allergy", "allergies",
    "phn", "dob", "age",
    "family", "social", "job", "occupation",
]


def looks_like_factual_data(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    if t.endswith("?"):
        return False
    if len(t) < 40:
        return False
    return any(k in t for k in FACT_KEYWORDS)


def promote_factual_text_to_background(context: SessionContext, text: str) -> bool:
    """
    Heuristic fact promotion used by /clinical_query.
    """
    if not looks_like_factual_data(text):
        _hydrate_identifiers_from_background_only(context)
        return False

    existing = ""
    try:
        existing = context.clinical_background.emr_dump or ""
    except Exception:
        existing = ""

    t = (text or "").strip()
    if not t:
        return False

    if t in existing:
        _hydrate_identifiers_from_background_only(context)
        return False

    try:
        context.clinical_background.emr_dump = (existing.rstrip() + "\n\n" + t) if existing else t
        context.session_meta.last_updated_at = _now_utc()
    except Exception:
        pass

    _hydrate_identifiers_from_background_only(context)
    return True


# =========================
# REFERRAL LETTER (PLAIN TEXT)
# =========================

def _build_re_line(context: SessionContext) -> str:
    ids = _extract_identifiers_from_background(context)

    parts = []
    if ids.get("name"):
        parts.append(ids["name"])
    if ids.get("dob"):
        parts.append(ids["dob"])
    if ids.get("phn"):
        parts.append(f"PHN: {ids['phn']}")

    if parts:
        return "Re: " + ", ".join(parts)
    return "Re: [Patient Name], [DOB], [PHN]"


def build_referral_prompt(context: SessionContext) -> str:
    re_line = _build_re_line(context)

    return f"""
You are a Canadian family medicine referral-letter assistant (Alberta).
Write a consultant referral letter in PLAIN TEXT.

HARD RULES:
- Output MUST be plain text only (no JSON).
- Do NOT invent demographics. If missing, leave them blank or keep placeholders.
- The letter MUST begin exactly with these first 2 lines (in this order):

Dear Colleague,
{re_line}

- Use clear headings and short paragraphs.
- Keep it clinically appropriate for a consultant.
- End the letter with EXACTLY:

Sincerely,

- Do NOT include any clinician name after "Sincerely,".

CONTENT SOURCES:
CLINICAL BACKGROUND:
{getattr(context.clinical_background, "emr_dump", None) or "None"}

VISIT TRANSCRIPT:
{getattr(context.transcript, "raw_text", None) or "None"}

Write the referral letter now.
""".strip()


def _ensure_re_line(letter_text: str, re_line: str) -> str:
    text = (letter_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    lines = text.split("\n")
    i0 = 0
    while i0 < len(lines) and not lines[i0].strip():
        i0 += 1
    if i0 >= len(lines):
        return text

    if lines[i0].strip() != "Dear Colleague,":
        return text

    i1 = i0 + 1
    while i1 < len(lines) and not lines[i1].strip():
        i1 += 1

    if i1 < len(lines) and lines[i1].strip().lower().startswith("re:"):
        return text

    insert_at = i0 + 1
    new_lines = lines[:insert_at] + [re_line, ""] + lines[insert_at:]
    return "\n".join(new_lines).strip()


def generate_referral_letter(context: SessionContext) -> str:
    re_line = _build_re_line(context)

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "Write a clean consultant referral letter. Plain text only."},
            {"role": "user", "content": build_referral_prompt(context)},
        ],
        temperature=0.2,
    )
    raw = (response.choices[0].message.content or "").strip()
    return _ensure_re_line(raw, re_line)


# =========================
# MAKE SOAP
# =========================

SOAP_STRUCTURE_MODEL = os.getenv("SOAP_STRUCTURE_MODEL", "gpt-4o-mini")
SOAP_SYNTHESIS_MODEL = os.getenv("SOAP_SYNTHESIS_MODEL", "gpt-4o")
SOAP_SCRUB_MODEL = os.getenv("SOAP_SCRUB_MODEL", "gpt-4o-mini")

SOAP_STRUCTURE_TEMP = 0.2
SOAP_SYNTHESIS_TEMP = 0.2
SOAP_SCRUB_TEMP = 0.0
SOAP_SCRUB_TOP_P = 1.0
SOAP_SCRUB_SEED = 42

SOAP_STRUCT_MAX_CHARS = 12000
SOAP_STRUCT_CHUNK_CHARS = 6000

SOAP_STRUCTURE_SYSTEM = """You are a clinical documentation engine for Canadian family medicine. Your job is to extract ONLY supported facts from the provided EncounterPacket and organize them into a structured JSON. Do not invent, assume, or add placeholders. You may group related facts into issue candidates if clearly supported by the transcript. Social history must come ONLY from today's transcript.

Return ONLY valid JSON matching this schema:
{ "issues": [...], "subjective_by_issue": {...}, "social_from_transcript": [...], "objective_facts": {...}, "procedures": {...}, "assessment_candidates": {...}, "plan_facts_by_issue": {...} }
""".strip()

SOAP_SYNTHESIS_SYSTEM = """You are an expert Canadian family physician producing a best-in-class SOAP note for family medicine. Use the EncounterPacket and StructuredExtraction. Do not invent. Do not include anything unsupported. You may perform controlled synthesis: group related symptoms into issues and state soft-inference diagnoses only if strongly supported by the facts. Avoid exhaustive ROS. Social Hx must be derived only from today's transcript.

Formatting rules:
- No commentary before or after.
- Bold section titles.
- No bullets.
- Each point on new line.
- Exactly one blank line between sections.
- Procedure section ONLY if performed.

Sections in order:
Issues:
Subjective:
Safety / Red Flags:
Social Hx:
Objective:
Assessment:
Procedure: (only if performed)
Plan:

Issues: numbered, minimal, grouped.

Medication rules:
- Use Canadian drug names/formulations.
- Include dosing only if explicitly present OR clinician clearly initiated therapy and dosing is standard in Canada; otherwise omit dosing.
""".strip()

SOAP_SCRUB_SYSTEM = """You are a strict clinical note auditor. Your task: remove anything not supported by the EncounterPacket or StructuredExtraction; remove filler; ensure Social Hx is transcript-only; ensure no "not mentioned" phrases; ensure format is EXACT (bold section titles, no bullets, each point new line, one blank line between sections). Output ONLY the final SOAP note text.
""".strip()


def _extract_emr_block(text: str, headers: List[str], max_lines: int = 30) -> str:
    if not text:
        return ""
    header_set = {h.lower().strip() for h in headers}
    lines = text.splitlines()
    start_indices: List[int] = []
    for i, line in enumerate(lines):
        norm = (line or "").strip().lower().rstrip(":")
        if norm in header_set:
            start_indices.append(i + 1)
    if not start_indices:
        return ""
    start = start_indices[-1]
    collected: List[str] = []
    for j in range(start, len(lines)):
        raw = (lines[j] or "").strip()
        if not raw:
            continue
        norm = raw.lower().rstrip(":")
        if norm in header_set:
            break
        if norm.endswith(":") and len(norm.split()) <= 4:
            break
        collected.append(raw)
        if len(collected) >= max_lines:
            break
    return "\n".join(collected).strip()


def _build_emr_context(emr_text: str) -> Dict[str, Any]:
    emr = (emr_text or "").strip()
    patient_banner = emr[:1200].strip() if emr else ""
    return {
        "patient_banner": patient_banner or None,
        "problem_list": _extract_emr_block(emr, ["Health Profile", "Problem List", "Diagnoses"]) or None,
        "meds": _extract_emr_block(emr, ["Medications", "Meds"]) or None,
        "allergies": _extract_emr_block(emr, ["Allergies", "Allergies/Intolerances"]) or None,
        "immunizations": _extract_emr_block(emr, ["Vaccines", "Immunizations"]) or None,
        "past_notes": _extract_emr_block(emr, ["Clinical Notes", "Patient Notes"]) or None,
    }


def _build_objective_data(emr_text: str, attachments_text: str = "") -> Dict[str, Any]:
    emr = (emr_text or "").strip()
    labs = _extract_emr_block(emr, ["Labs", "Laboratory", "Lab Results"])
    imaging = _extract_emr_block(emr, ["Investigations", "Imaging", "Diagnostic Imaging", "X-Ray", "Ultrasound"])
    vitals = _extract_emr_block(emr, ["Vitals", "Vital Signs"])
    exam = _extract_emr_block(emr, ["Objective Data", "Objective", "Exam", "Physical Exam"])
    poc = _extract_emr_block(emr, ["Point of Care", "POC", "Rapid Test"])

    att = (attachments_text or "").strip()
    if att:
        labs = (labs + ("\n\n" if labs else "") + att).strip()

    return {
        "vitals": vitals or None,
        "physical_exam": exam or None,
        "labs": labs or None,
        "imaging": imaging or None,
        "point_of_care_tests": poc or None,
    }


def build_encounter_packet(context: SessionContext, attachments_text: str = "") -> Dict[str, Any]:
    _hydrate_identifiers_best_effort(context)
    now = _now_utc().astimezone(ZoneInfo("America/Edmonton"))
    transcript_text = (getattr(context.transcript, "raw_text", None) or "").strip()
    emr_text = (getattr(context.clinical_background, "emr_dump", None) or "").strip()

    source = "dictation"
    if getattr(context.transcript, "segments", None):
        source = "ambient"
    elif getattr(context.transcript, "mode", None) == "imported":
        source = "typed"

    packet = {
        "metadata": {
            "encounter_id": context.session_meta.session_id,
            "clinician_name": None,
            "clinic_name": None,
            "timezone": "America/Edmonton",
            "encounter_datetime_iso": now.isoformat(),
            "visit_type": "unknown",
        },
        "transcript": {
            "source": source,
            "text": transcript_text,
            "speaker_map": None,
            "confidence": None,
        },
        "emr_context": _build_emr_context(emr_text),
        "objective_data": _build_objective_data(emr_text, attachments_text=attachments_text),
        "clinician_intent": {
            "note_style": "family_med_best_in_class",
            "allow_soft_inference": True,
            "strict_no_invention": True,
            "max_length": "medium",
        },
    }
    validate_encounter_packet(packet)
    return packet


def validate_encounter_packet(packet: Dict[str, Any]) -> None:
    required_top = ["metadata", "transcript", "emr_context", "objective_data", "clinician_intent"]
    for key in required_top:
        if key not in packet:
            raise ValueError(f"EncounterPacket missing '{key}'")
    meta = packet.get("metadata") or {}
    for key in ["encounter_id", "timezone", "encounter_datetime_iso", "visit_type"]:
        if key not in meta:
            raise ValueError(f"EncounterPacket.metadata missing '{key}'")
    transcript = packet.get("transcript") or {}
    if "text" not in transcript:
        raise ValueError("EncounterPacket.transcript missing 'text'")


def _chunk_text(text: str, max_chars: int) -> List[str]:
    t = (text or "").strip()
    if not t or len(t) <= max_chars:
        return [t]
    chunks: List[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        if end < len(t):
            cut = t.rfind("\n", start, end)
            if cut > start + 200:
                end = cut
        chunks.append(t[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _dedupe_list(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        raw = (item or "").strip()
        if not raw:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def normalize_structured_extraction(obj: Dict[str, Any]) -> Dict[str, Any]:
    data = obj if isinstance(obj, dict) else {}
    return {
        "issues": data.get("issues", []) or [],
        "subjective_by_issue": data.get("subjective_by_issue", {}) or {},
        "social_from_transcript": data.get("social_from_transcript", []) or [],
        "objective_facts": data.get("objective_facts", {}) or {},
        "procedures": data.get("procedures", {}) or {"performed": False, "description_facts_only": []},
        "assessment_candidates": data.get("assessment_candidates", {}) or {},
        "plan_facts_by_issue": data.get("plan_facts_by_issue", {}) or {},
    }


def merge_structured_extractions(extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = normalize_structured_extraction({})
    issue_map: Dict[str, Dict[str, Any]] = {}

    for ex in extractions:
        data = normalize_structured_extraction(ex)

        for issue in data.get("issues", []) or []:
            title = (issue.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key not in issue_map:
                issue_map[key] = {
                    "title": title,
                    "supporting_quotes": _dedupe_list(issue.get("supporting_quotes", []) or []),
                    "supporting_facts": _dedupe_list(issue.get("supporting_facts", []) or []),
                }
            else:
                issue_map[key]["supporting_quotes"] = _dedupe_list(
                    issue_map[key]["supporting_quotes"] + (issue.get("supporting_quotes", []) or [])
                )
                issue_map[key]["supporting_facts"] = _dedupe_list(
                    issue_map[key]["supporting_facts"] + (issue.get("supporting_facts", []) or [])
                )

        for k, v in (data.get("subjective_by_issue", {}) or {}).items():
            if not isinstance(v, list):
                continue
            merged.setdefault("subjective_by_issue", {})
            merged["subjective_by_issue"][k] = _dedupe_list(
                (merged["subjective_by_issue"].get(k, []) or []) + v
            )

        merged["social_from_transcript"] = _dedupe_list(
            (merged.get("social_from_transcript", []) or []) + (data.get("social_from_transcript", []) or [])
        )

        for key in ["vitals", "physical_exam", "labs", "imaging", "point_of_care_tests"]:
            items = data.get("objective_facts", {}).get(key, []) or []
            merged.setdefault("objective_facts", {})
            merged["objective_facts"][key] = _dedupe_list(
                (merged["objective_facts"].get(key, []) or []) + items
            )

        if data.get("procedures", {}).get("performed"):
            merged.setdefault("procedures", {"performed": False, "description_facts_only": []})
            merged["procedures"]["performed"] = True
            merged["procedures"]["description_facts_only"] = _dedupe_list(
                (merged["procedures"].get("description_facts_only", []) or []) +
                (data.get("procedures", {}).get("description_facts_only", []) or [])
            )

        for key in ["dx_explicit", "dx_inferred_soft", "reasoning_facts"]:
            items = data.get("assessment_candidates", {}).get(key, []) or []
            merged.setdefault("assessment_candidates", {})
            merged["assessment_candidates"][key] = _dedupe_list(
                (merged["assessment_candidates"].get(key, []) or []) + items
            )

        for k, v in (data.get("plan_facts_by_issue", {}) or {}).items():
            if not isinstance(v, list):
                continue
            merged.setdefault("plan_facts_by_issue", {})
            merged["plan_facts_by_issue"][k] = _dedupe_list(
                (merged["plan_facts_by_issue"].get(k, []) or []) + v
            )

    merged["issues"] = list(issue_map.values())
    return merged


def _run_structure_extraction(packet: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(packet, ensure_ascii=False)
    response = client.chat.completions.create(
        model=SOAP_STRUCTURE_MODEL,
        messages=[
            {"role": "system", "content": SOAP_STRUCTURE_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=SOAP_STRUCTURE_TEMP,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _run_structure_extraction_chunked(packet: Dict[str, Any]) -> Dict[str, Any]:
    transcript = (packet.get("transcript", {}) or {}).get("text", "") or ""
    if len(transcript) <= SOAP_STRUCT_MAX_CHARS:
        return normalize_structured_extraction(_run_structure_extraction(packet))

    chunks = _chunk_text(transcript, SOAP_STRUCT_CHUNK_CHARS)
    results: List[Dict[str, Any]] = []
    for chunk in chunks:
        packet_copy = json.loads(json.dumps(packet))
        packet_copy["transcript"]["text"] = chunk
        results.append(_run_structure_extraction(packet_copy))
    return normalize_structured_extraction(merge_structured_extractions(results))


def _run_soap_synthesis(packet: Dict[str, Any], extraction: Dict[str, Any]) -> str:
    user_content = (
        "EncounterPacket:\n"
        + json.dumps(packet, ensure_ascii=False)
        + "\n\nStructuredExtraction:\n"
        + json.dumps(extraction, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model=SOAP_SYNTHESIS_MODEL,
        messages=[
            {"role": "system", "content": SOAP_SYNTHESIS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=SOAP_SYNTHESIS_TEMP,
    )
    return (response.choices[0].message.content or "").strip()


def _run_soap_scrub(packet: Dict[str, Any], extraction: Dict[str, Any], draft: str) -> str:
    user_content = (
        "EncounterPacket:\n"
        + json.dumps(packet, ensure_ascii=False)
        + "\n\nStructuredExtraction:\n"
        + json.dumps(extraction, ensure_ascii=False)
        + "\n\nDraftSOAP:\n"
        + (draft or "")
    )
    response = _chat_complete_with_seed(
        model=SOAP_SCRUB_MODEL,
        messages=[
            {"role": "system", "content": SOAP_SCRUB_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=SOAP_SCRUB_TEMP,
        top_p=SOAP_SCRUB_TOP_P,
        seed=SOAP_SCRUB_SEED,
    )
    return (response.choices[0].message.content or "").strip()


def _chat_complete_with_seed(**kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "seed" in msg:
            kwargs.pop("seed", None)
            return client.chat.completions.create(**kwargs)
        raise


def _strip_bullets(line: str) -> str:
    raw = (line or "").lstrip()
    if raw.startswith("- "):
        return raw[2:].strip()
    if raw.startswith("• "):
        return raw[2:].strip()
    if raw.startswith("•"):
        return raw[1:].strip()
    return line.strip()


def should_include_procedure_section(extraction: Dict[str, Any]) -> bool:
    proc = (extraction or {}).get("procedures", {}) or {}
    return bool(proc.get("performed"))


def normalize_soap_output(text: str, include_procedure: bool) -> str:
    headers = [
        "Issues",
        "Subjective",
        "Safety / Red Flags",
        "Social Hx",
        "Objective",
        "Assessment",
        "Procedure",
        "Plan",
    ]
    header_pattern = re.compile(r"^\s*\*{0,2}\s*(" + "|".join(re.escape(h) for h in headers) + r")\s*:?\s*\*{0,2}\s*$")
    sections: Dict[str, List[str]] = {h: [] for h in headers}
    current = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = header_pattern.match(line)
        if match:
            current = match.group(1)
            continue
        if current:
            cleaned = _strip_bullets(line)
            if cleaned:
                sections[current].append(cleaned)

    out_sections: List[str] = []
    for header in headers:
        if header == "Procedure" and not include_procedure:
            continue
        lines = sections.get(header, []) or []
        if header == "Safety / Red Flags" and not lines:
            lines = ["none"]
        section_lines = [f"**{header}:**"]
        if lines:
            section_lines.extend(lines)
        out_sections.append("\n".join(section_lines))

    return "\n\n".join(out_sections).strip()


def validate_soap_output(text: str, include_procedure: bool) -> bool:
    lines = (text or "").splitlines()
    for line in lines:
        raw = line.lstrip()
        if raw.startswith("- ") or raw.startswith("•"):
            return False

    headers = [
        "Issues",
        "Subjective",
        "Safety / Red Flags",
        "Social Hx",
        "Objective",
        "Assessment",
        "Plan",
    ]
    if include_procedure:
        headers.insert(6, "Procedure")

    found = [line.strip() for line in lines if line.strip().startswith("**") and line.strip().endswith("**")]
    expected = [f"**{h}:**" for h in headers]
    for h in expected:
        if h not in found:
            return False
    return True


def make_soap(context: SessionContext, attachments_text: str = "") -> dict:
    packet = build_encounter_packet(context, attachments_text=attachments_text)
    extraction = _run_structure_extraction_chunked(packet)
    draft_text = _run_soap_synthesis(packet, extraction)
    scrubbed_text = _run_soap_scrub(packet, extraction, draft_text)

    final_text = normalize_soap_output(
        scrubbed_text,
        include_procedure=should_include_procedure_section(extraction),
    )
    if not validate_soap_output(final_text, include_procedure=should_include_procedure_section(extraction)):
        logger.warning("SOAP format validation failed; returning normalized output.")
    provenance = {
        "issues": extraction.get("issues", []),
        "plan_facts_by_issue": extraction.get("plan_facts_by_issue", {}),
    }

    return {
        "soap_text": final_text,
        "generated_at": _now_utc(),
        "context_hash": _hash_context(context),
        "provenance": provenance,
    }


# =========================
# DIFFERENTIAL COACH (SUCCINCT + HARD MAX)
# =========================

MIN_TRANSCRIPT_CHARS = 60
MIN_TRANSCRIPT_WORDS = 12

COACH_MAX_TRANSCRIPT_CHARS = 4200
COACH_MAX_BACKGROUND_CHARS = 4200

DIFFERENTIAL_MODEL = "gpt-5.2"

DDX_MAX = 4
CANT_MISS_MAX = 4
KEY_Q_MAX = 6
PLAN_MAX = 6


def _has_meaningful_transcript(context: SessionContext) -> bool:
    t = (getattr(context.transcript, "raw_text", None) or "").strip()
    if len(t) < MIN_TRANSCRIPT_CHARS:
        return False
    if len(t.split()) < MIN_TRANSCRIPT_WORDS:
        return False
    return True


def _get_demographics_block(context: SessionContext) -> Dict[str, Any]:
    ids = _extract_identifiers_from_background(context)

    patient = _get_patient_obj(context)
    if patient is None:
        return {
            "name": ids.get("name", ""),
            "age": ids.get("age", None),
            "sex": ids.get("sex", ""),
            "dob": ids.get("dob", ""),
            "phn": ids.get("phn", ""),
        }

    name = _get_str_attr(patient, "name") or ids.get("name", "")
    dob = _get_str_attr(patient, "dob") or ids.get("dob", "")
    phn = _get_str_attr(patient, "phn") or ids.get("phn", "")
    sex_val = _get_str_attr(patient, "sex") or _get_str_attr(patient, "gender") or ids.get("sex", "")

    age = getattr(patient, "age", None)
    try:
        age = int(age) if age is not None and str(age).strip() != "" else ids.get("age", None)
    except Exception:
        age = ids.get("age", None)

    return {"name": name, "age": age, "sex": sex_val, "dob": dob, "phn": phn}


def build_coach_context(context: SessionContext) -> Dict[str, Any]:
    transcript_raw = (getattr(context.transcript, "raw_text", None) or "").strip()
    background_raw = (getattr(context.clinical_background, "emr_dump", None) or "").strip()

    coach_ctx = {
        "demographics": _get_demographics_block(context),
        "transcript": _clip_text(transcript_raw, COACH_MAX_TRANSCRIPT_CHARS) if transcript_raw else "",
        "clinical_background": _clip_text(background_raw, COACH_MAX_BACKGROUND_CHARS) if background_raw else "",
        "stats": {
            "transcript_chars": len(transcript_raw),
            "background_chars": len(background_raw),
        },
    }
    return coach_ctx


def _coach_output_schema() -> str:
    return f"""
Return STRICT JSON with keys exactly:
ddx
cant_miss_questions
key_questions_to_refine_ddx
suggested_plan
confidence

Hard rules:
- Every list item MUST be a plain string (no dict/object items).
- Ddx must be ordered most important/likely → least.
- cant_miss_questions and key_questions_to_refine_ddx MUST be written as questions ending with "?".
- Keep it succinct for live use (short phrases, avoid long clauses).
- Each item should be <= 12 words when possible.

Hard maxima:
- ddx: max {DDX_MAX}
- cant_miss_questions: max {CANT_MISS_MAX}
- key_questions_to_refine_ddx: max {KEY_Q_MAX}
- suggested_plan: max {PLAN_MAX}
- confidence: one short line
""".strip()


def build_differential_prompt(context: SessionContext, web_context: str = "") -> str:
    coach_ctx = build_coach_context(context)
    web_block = web_context.strip() or "None"

    return f"""
You are a clinical reasoning assistant supporting a licensed clinician.

PRIORITY RULES:
1) VISIT TRANSCRIPT is the primary source for today's presentation.
2) CLINICAL BACKGROUND is secondary: use only if relevant.
3) Do NOT generate a generic problem-list differential; focus on active concern(s).
4) Do NOT invent missing data.

CONTEXT:
DEMOGRAPHICS (if known):
{json.dumps(coach_ctx.get("demographics", {}), ensure_ascii=False, indent=2)}

CLINICAL BACKGROUND (secondary):
{coach_ctx.get("clinical_background") or "None"}

VISIT TRANSCRIPT (primary):
{coach_ctx.get("transcript") or "None"}

WEB SOURCES (unverified; use for guidelines, cite URLs if used):
{web_block}

OUTPUT RULES:
- Return STRICT JSON ONLY. No markdown. No extra text.
- suggested_plan must include pharmacotherapy suggestions with dose + duration when appropriate.
- Only suggest drugs/doses that are available in Alberta; if unsure, say "Verify Alberta availability" and avoid specifics.
- If suggesting antibiotics, follow Bugs & Drugs (Alberta) guidance; if unsure, say "Verify Bugs & Drugs" and avoid specifics.

{_coach_output_schema()}
""".strip()


def _fallback_waiting_payload(reason: str) -> str:
    waiting = {
        "ddx": [],
        "cant_miss_questions": ["What is the chief complaint and when did it start?"],
        "key_questions_to_refine_ddx": ["What are the key associated symptoms and red flags?"],
        "suggested_plan": ["Start recording or paste the visit details, then rerun Clinical Coach."],
        "confidence": reason,
        "meta": {
            "generated_at": _now_utc().isoformat(),
            "model": DIFFERENTIAL_MODEL,
            "note": "Coach did not run due to insufficient transcript.",
        },
    }
    return json.dumps(waiting, ensure_ascii=False, indent=2)


def _truncate_words(text: str, max_words: int) -> str:
    if not text or not max_words or max_words <= 0:
        return text
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words]).rstrip(".,;:") + "..."


def _coerce_str_list(x, max_n: int, ensure_question: bool = False, max_words: int = 0) -> List[str]:
    if x is None:
        items = []
    elif isinstance(x, list):
        items = x
    else:
        items = [x]

    out: List[str] = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, str):
            s = it.strip()
        elif isinstance(it, dict):
            s = (it.get("diagnosis") or it.get("text") or "").strip()
            if not s:
                try:
                    s = json.dumps(it, ensure_ascii=False)
                except Exception:
                    s = str(it).strip()
        else:
            s = str(it).strip()

        if not s:
            continue

        if max_words and max_words > 0:
            s = _truncate_words(s, max_words)

        if ensure_question and not s.endswith("?"):
            s = s.rstrip(".")
            s = s + "?"

        out.append(s)
        if len(out) >= max_n:
            break

    return out


def run_differential_coach(context: SessionContext) -> str:
    if not _has_meaningful_transcript(context):
        return _fallback_waiting_payload("Insufficient information: live transcript not yet available.")

    web_context = ""
    if WEB_SEARCH_ENABLED:
        try:
            web_query = _build_web_query_for_coach(context)
            results = _web_search(web_query)
            web_context = _format_web_results(results)
        except Exception:
            web_context = ""

    prompt = build_differential_prompt(context, web_context=web_context)
    coach_ctx = build_coach_context(context)

    response_text = ""
    used_response_format = False
    try:
        resp = client.chat.completions.create(
            model=DIFFERENTIAL_MODEL,
            messages=[
                {"role": "system", "content": "Clinical reasoning only. Transcript-first. Return STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        used_response_format = True
        response_text = (resp.choices[0].message.content or "").strip()
    except Exception:
        resp = client.chat.completions.create(
            model=DIFFERENTIAL_MODEL,
            messages=[
                {"role": "system", "content": "Clinical reasoning only. Transcript-first. Return STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        response_text = (resp.choices[0].message.content or "").strip()

    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(response_text)
        if not isinstance(parsed, dict):
            raise ValueError("Coach JSON was not an object")
    except Exception:
        parsed = {
            "ddx": [],
            "cant_miss_questions": ["Do you have enough detail to screen for red flags?"],
            "key_questions_to_refine_ddx": ["What key history or exam findings are missing?"],
            "suggested_plan": ["Rerun after more transcript is available."],
            "confidence": "Low (model returned non-JSON output).",
            "meta": {"raw_model_output": response_text[:2000] + ("..." if len(response_text) > 2000 else "")},
        }

    ddx = _coerce_str_list(parsed.get("ddx"), DDX_MAX, ensure_question=False, max_words=10)
    cant_miss = _coerce_str_list(parsed.get("cant_miss_questions"), CANT_MISS_MAX, ensure_question=True, max_words=12)
    key_q = _coerce_str_list(parsed.get("key_questions_to_refine_ddx"), KEY_Q_MAX, ensure_question=True, max_words=12)
    plan = _coerce_str_list(parsed.get("suggested_plan"), PLAN_MAX, ensure_question=False, max_words=14)
    confidence = (parsed.get("confidence") or "").strip()
    if confidence:
        confidence = re.sub(r"\s+", " ", confidence).strip()

    out: Dict[str, Any] = {
        "ddx": ddx,
        "cant_miss_questions": cant_miss,
        "key_questions_to_refine_ddx": key_q,
        "suggested_plan": plan,
        "confidence": confidence,
    }

    meta = parsed.get("meta", {})
    if not isinstance(meta, dict):
        meta = {"note": str(meta)}

    meta.update({
        "generated_at": _now_utc().isoformat(),
        "model": DIFFERENTIAL_MODEL,
        "used_response_format_json": used_response_format,
        "context_used": {
            "demographics_present": bool(
                (coach_ctx.get("demographics") or {}).get("name")
                or (coach_ctx.get("demographics") or {}).get("age")
                or (coach_ctx.get("demographics") or {}).get("sex")
                or (coach_ctx.get("demographics") or {}).get("phn")
            ),
            "background_present": bool((coach_ctx.get("clinical_background") or "").strip()),
            "transcript_present": bool((coach_ctx.get("transcript") or "").strip()),
        },
        "prompt_version": "coach_v3_succinct_hardcaps",
    })
    out["meta"] = meta

    return json.dumps(out, ensure_ascii=False, indent=2)


# =========================
# CLINICAL QUERY (SUPPORTS ATTACHMENTS TEXT + IMAGE ATTACHMENTS)
# =========================

DESCRIPTIVE_PATTERNS = [
    r"\bwhat (meds|medications)\b",
    r"\bwhat is the patient on\b",
    r"\blist (meds|medications)\b",
    r"\bcurrent medications\b",
    r"\blist (his|her|their)?\s*(main )?(issues|problems|dx|diagnoses)\b",
    r"\bmain issues\b",
    r"\bproblem list\b",
    r"\bwhat is this (pdf|document|file|attachment)\b",
    r"\bwhat does this (pdf|document|file|attachment) say\b",
    r"\bsummarize (this|the) (pdf|document|file|attachment)\b",
    r"\breview (this|the) (pdf|document|file|attachment)\b",
    r"\bextract (text|contents?)\b",
    r"\bwhat is this about\b",
    r"\bcan you read this\b",
    r"\bdescribe (this|the)?\s*image\b",
    r"\bdescribe (this|the)?\s*lesion\b",
    r"\bdermatolog(y|ist)\b",
    r"\bwhat do you see\b",
    r"\bimage description\b",
]


def is_descriptive_query(query: str) -> bool:
    q = (query or "").lower()
    return any(re.search(p, q) for p in DESCRIPTIVE_PATTERNS)


def _infer_query_format(query: str) -> Tuple[str, bool]:
    q = (query or "").lower()
    wants_plan = any(k in q for k in [
        "plan", "treat", "treatment", "manage", "management",
        "moving forward", "next step", "next steps", "what to do", "recommend"
    ])
    wants_summary = any(k in q for k in ["summarize", "summary", "overview", "snapshot", "brief"])
    wants_list = any(k in q for k in [
        "list", "issues", "problem", "problems", "dx", "diagnos", "main issues", "active issues"
    ])

    if wants_list and not wants_plan and not wants_summary:
        return "list_only", True
    if wants_plan and wants_summary:
        return "summary_and_plan", False
    if wants_plan:
        return "plan_only", False
    if wants_summary:
        return "summary_only", False
    return "concise", False


CLINICAL_QUERY_MAX_BACKGROUND_CHARS = 12000
CLINICAL_QUERY_MAX_TRANSCRIPT_CHARS = 9000
CLINICAL_QUERY_MAX_ATTACHMENTS_CHARS = 12000


def _format_attachments_block(attachments_text: str) -> str:
    t = (attachments_text or "").strip()
    if not t:
        return "None"
    return _clip_text(t, CLINICAL_QUERY_MAX_ATTACHMENTS_CHARS)


def build_clinical_query_prompt(
    context: SessionContext,
    query: str,
    expand: bool,
    descriptive: bool,
    format_hint: str,
    attachments_text: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    web_context: str = "",
    kb_context: str = "",
    guideline_context: str = "",
) -> str:
    _hydrate_identifiers_best_effort(context)

    bg = (getattr(context.clinical_background, "emr_dump", None) or "").strip()
    tx = (getattr(context.transcript, "raw_text", None) or "").strip()

    bg_clip = _clip_text(bg, CLINICAL_QUERY_MAX_BACKGROUND_CHARS) if bg else "None"
    tx_clip = _clip_text(tx, CLINICAL_QUERY_MAX_TRANSCRIPT_CHARS) if tx else "None"
    att_block = _format_attachments_block(attachments_text)
    images_summary = _attachments_image_summary(attachments)
    web_block = web_context.strip() or "None"
    kb_block = kb_context.strip() or "None"
    guideline_block = guideline_context.strip() or "None"

    return f"""
You are Centaur: a focused, formal, exacting AI consultant. Your purpose is to deliver high-value, correct, practical outputs with minimal fluff.

Jurisdiction: Alberta, Canada.
Audience: licensed physician.
This output is NOT for charting.

CORE BEHAVIOR:
- Be direct, structured, and thorough. Avoid preambles, hype, emojis, or filler.
- Keep it concise: short sentences, minimal clauses, no long paragraphs.
- Do not repeat the question. Do not praise the user.
- Show, don’t tell: provide the actual artifact (steps, checklist, template, recommendation).
- If information is missing, make the smallest reasonable assumptions that do NOT add patient facts; state them plainly.
- If ambiguous, ask at most 1–2 targeted clarifying questions at the end; otherwise provide a best-effort answer plus options.
- If multiple valid approaches exist and the format allows, present 2–3 options with tradeoffs and a recommendation.

REASONING & ACCURACY:
- Prioritize correctness over speed. If a claim is uncertain, say so and suggest how to verify.
- Do NOT reveal chain-of-thought or private reasoning; provide concise rationale only as needed.
- For medical/legal/financial topics, include appropriate cautions and suggest professional verification where relevant.

SAFETY & PRIVACY:
- Refuse requests that meaningfully facilitate wrongdoing or unsafe behavior; provide safe alternatives.
- Respect privacy; do not request sensitive personal data unless strictly necessary, and then minimize what you collect.

HARD RULES:
- Do NOT invent patient data.
- Be explicit about uncertainty.
- Use Canadian drug names and doses.
- Avoid teaching language.
- If the question is asking about an ATTACHED DOCUMENT, do NOT claim to have seen it unless its extracted text is present below.
- If the user asks to describe an image/lesion, only do so if image(s) are present (see ATTACHED IMAGES).
- If LOCAL KNOWLEDGE BASE content is provided and relevant, prioritize it over WEB SOURCES.
- If GUIDELINE RUNNER output is provided, use it as the authoritative decision path and do NOT invent additional steps.
- When using GUIDELINE RUNNER output, surface: decision points, next steps, missing inputs, and cite source_url/page/node label from evidence_spans.

IMPORTANT LOGIC RULE:
- If the question is DESCRIPTIVE (e.g. listing meds/history/facts OR summarizing an attached document OR describing an image):
  - recommended_regimens MUST be an empty array []
  - Do NOT suggest new treatments unless explicitly asked
- Only populate recommended_regimens when the clinician is ASKING FOR TREATMENT

QUESTION TYPE:
{"DESCRIPTIVE" if descriptive else "CLINICAL DECISION"}

DEPTH:
{"Expanded reasoning allowed" if expand else "Brief and practical"}

CLINICAL BACKGROUND (source of truth):
{bg_clip}

VISIT TRANSCRIPT:
{tx_clip}

ATTACHMENTS (extracted text; may be clipped):
{att_block}

ATTACHED IMAGES (provided in the message payload, if any):
{images_summary}

WEB SOURCES (unverified; use when freshness matters; cite URLs if used):
{web_block}

LOCAL KNOWLEDGE BASE (priority; cite URLs if used):
{kb_block}

GUIDELINE RUNNER (deterministic; cite URLs/page/node if used):
{guideline_block}

QUESTION:
{query}

Return STRICT JSON ONLY with keys:
direct_answer
recommended_regimens
interactions_and_contraindications
tests_or_workup
red_flags
missing_critical_info
uncertainties
follow_up

DIRECT_ANSWER FORMAT (match the question exactly; no extra sections):
- format_hint = "{format_hint}"
- If format_hint="list_only": return 3–8 bullet points only. No headings or extra sections.
- If format_hint="summary_only": return a short paragraph (2–5 sentences). No headings.
- If format_hint="plan_only": include a "Plan" heading with concise bullets.
- If format_hint="summary_and_plan": use two headings: "What we know" and "Plan moving forward", with bullets.
- If format_hint="concise": 1–2 short paragraphs, no headings.

If DESCRIPTIVE:
- recommended_regimens = []
- tests_or_workup = []
 - leave red_flags, missing_critical_info, uncertainties, follow_up empty unless explicitly asked
""".strip()


def run_clinical_query(
    context: SessionContext,
    query: str,
    mode: str = "brief",
    attachments_text: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    model_override: Optional[str] = None,
) -> str:
    _hydrate_identifiers_best_effort(context)

    q_lower = (query or "").lower()
    mode_norm = (mode or "").strip().lower()
    fast_mode = True
    expand = ("expand" in q_lower) or (mode_norm == "expand")
    format_hint, force_descriptive = _infer_query_format(query)
    descriptive = is_descriptive_query(query) or force_descriptive

    kb_context = ""
    guideline_context = ""
    if KB_ENABLED:
        try:
            kb_results = search_kb(query, limit=5)
            kb_context = format_kb_context(kb_results)
        except Exception:
            kb_context = ""
        try:
            bg_text = (getattr(context.clinical_background, "emr_dump", "") or "").strip()
            tx_text = (getattr(context.transcript, "raw_text", "") or "").strip()
            guideline_result = run_guideline_runner(query, f"{bg_text}\n{tx_text}".strip())
            if guideline_result:
                guideline_context = json.dumps(guideline_result, ensure_ascii=True)
        except Exception:
            guideline_context = ""

    web_context = ""
    if not kb_context and not guideline_context and WEB_SEARCH_ENABLED and CLINICAL_QUERY_USE_WEB:
        try:
            web_query = _build_web_query_for_clinical_query(context, query)
            results = _web_search(web_query)
            web_context = _format_web_results(results)
        except Exception:
            web_context = ""

    prompt = build_clinical_query_prompt(
        context=context,
        query=query,
        expand=expand,
        descriptive=descriptive,
        format_hint=format_hint,
        attachments_text=attachments_text,
        attachments=attachments,
        web_context=web_context,
        kb_context=kb_context,
        guideline_context=guideline_context,
    )

    has_images = False
    if attachments:
        for a in attachments:
            if (
                isinstance(a, dict)
                and (a.get("kind") or "").lower().strip() == "image"
                and (a.get("data_base64") or "").strip()
            ):
                has_images = True
                break

    user_content = _build_multimodal_user_content(prompt, attachments if has_images else None)
    text_model = (model_override or CLINICAL_QUERY_TEXT_MODEL).strip()
    vision_model = (model_override or CLINICAL_QUERY_VISION_MODEL).strip()
    model = vision_model if (has_images and isinstance(user_content, list)) else text_model

    def _unique_models(models: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for m in models:
            m = (m or "").strip()
            if not m or m in seen:
                continue
            seen.add(m)
            out.append(m)
        return out

    if has_images:
        candidates = _unique_models([
            model,
            CLINICAL_QUERY_VISION_MODEL,
            "gpt-5.2",
            "gpt-4o-mini",
        ])
    else:
        candidates = _unique_models([
            model,
            CLINICAL_QUERY_TEXT_MODEL,
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-4o-mini",
            "gpt-5.2",
        ])

    system_msg = (
        "You are Centaur: a focused, formal, exacting AI consultant for licensed clinicians in Alberta. "
        "Conservative Alberta-appropriate advice. "
        "Do not reveal chain-of-thought. "
        "If PDF extracted text is provided, use it; otherwise say it is missing. "
        "If image(s) are provided in the message payload, you may describe their visible features."
    )

    temperature = None if fast_mode else 0.2
    base_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": CLINICAL_QUERY_MAX_TOKENS,
    }
    if CLINICAL_QUERY_SERVICE_TIER:
        base_kwargs["service_tier"] = CLINICAL_QUERY_SERVICE_TIER
    if temperature is not None:
        base_kwargs["temperature"] = temperature

    last_error: Optional[Exception] = None

    def _attempt(use_response_format: bool) -> Optional[str]:
        nonlocal last_error
        for cand in candidates:
            try:
                base_kwargs["model"] = cand
                if use_response_format:
                    response = client.chat.completions.create(
                        **base_kwargs,
                        response_format={"type": "json_object"},
                    )
                else:
                    response = client.chat.completions.create(**base_kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = e
                continue
        return None

    result = _attempt(True)
    if result is not None:
        return result

    result = _attempt(False)
    if result is not None:
        return result

    if "service_tier" in base_kwargs:
        base_kwargs.pop("service_tier", None)
        result = _attempt(True)
        if result is not None:
            return result
        result = _attempt(False)
        if result is not None:
            return result
    err = {
        "direct_answer": "",
        "recommended_regimens": [],
        "interactions_and_contraindications": [],
        "tests_or_workup": [],
        "red_flags": [],
        "missing_critical_info": ["Model call failed."],
        "uncertainties": [str(last_error) if last_error else "Unknown error"],
        "follow_up": ["Retry the query. If using images, confirm attachments are being uploaded and passed to /clinical_query."],
        "meta": {
            "model_attempted": candidates[-1] if candidates else model,
            "has_images": has_images,
            "attachments_text_present": bool((attachments_text or "").strip()),
            "generated_at": _now_utc().isoformat(),
        },
    }
    return json.dumps(err, ensure_ascii=False, indent=2)


def run_clinical_query_stream(
    context: SessionContext,
    query: str,
    mode: str = "brief",
    attachments_text: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    model_override: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream tokens from the clinical query response.
    Yields plain text chunks.
    """
    _hydrate_identifiers_best_effort(context)

    q_lower = (query or "").lower()
    mode_norm = (mode or "").strip().lower()
    fast_mode = True
    expand = ("expand" in q_lower) or (mode_norm == "expand")
    format_hint, force_descriptive = _infer_query_format(query)
    descriptive = is_descriptive_query(query) or force_descriptive

    kb_context = ""
    guideline_context = ""
    if KB_ENABLED:
        try:
            kb_results = search_kb(query, limit=5)
            kb_context = format_kb_context(kb_results)
        except Exception:
            kb_context = ""
        try:
            bg_text = (getattr(context.clinical_background, "emr_dump", "") or "").strip()
            tx_text = (getattr(context.transcript, "raw_text", "") or "").strip()
            guideline_result = run_guideline_runner(query, f"{bg_text}\n{tx_text}".strip())
            if guideline_result:
                guideline_context = json.dumps(guideline_result, ensure_ascii=True)
        except Exception:
            guideline_context = ""

    web_context = ""
    if not kb_context and not guideline_context and WEB_SEARCH_ENABLED and CLINICAL_QUERY_USE_WEB:
        try:
            web_query = _build_web_query_for_clinical_query(context, query)
            results = _web_search(web_query)
            web_context = _format_web_results(results)
        except Exception:
            web_context = ""

    prompt = build_clinical_query_prompt(
        context=context,
        query=query,
        expand=expand,
        descriptive=descriptive,
        format_hint=format_hint,
        attachments_text=attachments_text,
        attachments=attachments,
        web_context=web_context,
        kb_context=kb_context,
        guideline_context=guideline_context,
    )

    has_images = False
    if attachments:
        for a in attachments:
            if (
                isinstance(a, dict)
                and (a.get("kind") or "").lower().strip() == "image"
                and (a.get("data_base64") or "").strip()
            ):
                has_images = True
                break

    user_content = _build_multimodal_user_content(prompt, attachments if has_images else None)
    text_model = (model_override or CLINICAL_QUERY_TEXT_MODEL).strip()
    vision_model = (model_override or CLINICAL_QUERY_VISION_MODEL).strip()
    model = vision_model if (has_images and isinstance(user_content, list)) else text_model

    def _unique_models(models: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for m in models:
            m = (m or "").strip()
            if not m or m in seen:
                continue
            seen.add(m)
            out.append(m)
        return out

    if has_images:
        candidates = _unique_models([
            model,
            CLINICAL_QUERY_VISION_MODEL,
            "gpt-5.2",
            "gpt-4o-mini",
        ])
    else:
        candidates = _unique_models([
            model,
            CLINICAL_QUERY_TEXT_MODEL,
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-4o-mini",
            "gpt-5.2",
        ])

    system_msg = (
        "You are Centaur: a focused, formal, exacting AI consultant for licensed clinicians in Alberta. "
        "Conservative Alberta-appropriate advice. "
        "Do not reveal chain-of-thought. "
        "If PDF extracted text is provided, use it; otherwise say it is missing. "
        "If image(s) are provided in the message payload, you may describe their visible features."
    )

    temperature = None if fast_mode else 0.2
    base_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": CLINICAL_QUERY_MAX_TOKENS,
    }
    if CLINICAL_QUERY_SERVICE_TIER:
        base_kwargs["service_tier"] = CLINICAL_QUERY_SERVICE_TIER
    if temperature is not None:
        base_kwargs["temperature"] = temperature

    for cand in candidates:
        try:
            base_kwargs["model"] = cand
            response = client.chat.completions.create(
                **base_kwargs,
                response_format={"type": "json_object"},
                stream=True,
            )

            for chunk in response:
                try:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                except Exception:
                    piece = None
                if piece:
                    yield piece
            return
        except Exception:
            continue

    for cand in candidates:
        try:
            base_kwargs["model"] = cand
            response = client.chat.completions.create(
                **base_kwargs,
                stream=True,
            )
            for chunk in response:
                try:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None)
                except Exception:
                    piece = None
                if piece:
                    yield piece
            return
        except Exception:
            continue

    if "service_tier" in base_kwargs:
        base_kwargs.pop("service_tier", None)
        for cand in candidates:
            try:
                base_kwargs["model"] = cand
                response = client.chat.completions.create(
                    **base_kwargs,
                    response_format={"type": "json_object"},
                    stream=True,
                )
                for chunk in response:
                    try:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, "content", None)
                    except Exception:
                        piece = None
                    if piece:
                        yield piece
                return
            except Exception:
                continue

        for cand in candidates:
            try:
                base_kwargs["model"] = cand
                response = client.chat.completions.create(
                    **base_kwargs,
                    stream=True,
                )
                for chunk in response:
                    try:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, "content", None)
                    except Exception:
                        piece = None
                    if piece:
                        yield piece
                return
            except Exception:
                continue

    # Final fallback: return non-streaming output as a single chunk
    yield run_clinical_query(
        context=context,
        query=query,
        mode=mode,
        attachments_text=attachments_text,
        attachments=attachments,
        model_override=model_override,
    )

# =========================
# Patient Summary (EMR-only)
# =========================

_SUMMARY_HEADERS = {
    "family hx",
    "family history",
    "social history",
    "social hx",
    "health profile",
    "problem list",
    "diagnoses",
    "allergies",
    "allergies/intolerances",
    "medications",
    "clinical notes",
    "patient notes",
    "notes",
    "active & past medical conditions",
    "active medical conditions",
    "surgical / procedural history",
    "surgical history",
    "review of systems",
    "objective data",
    "assessment",
    "plan",
    "subjective",
    "objective",
    "soap",
    "labs",
    "investigations",
    "imaging",
    "letters",
}


def _summary_norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _summary_is_header(line: str) -> bool:
    raw = _summary_norm_line(line).lower().rstrip(":")
    if not raw:
        return False
    if raw in _SUMMARY_HEADERS:
        return True
    for h in _SUMMARY_HEADERS:
        if raw.startswith(h + " "):
            return True
    return False


def _extract_section_block_last(text: str, headers: List[str], max_lines: int = 12) -> str:
    if not text:
        return ""
    header_set = {h.lower() for h in headers}
    lines = text.splitlines()
    matches: List[int] = []
    for i, line in enumerate(lines):
        norm = _summary_norm_line(line).lower().rstrip(":")
        if norm in header_set:
            matches.append(i + 1)
    if not matches:
        return ""
    start_idx = matches[-1]
    collected: List[str] = []
    for j in range(start_idx, len(lines)):
        line = _summary_norm_line(lines[j])
        if not line:
            continue
        if _summary_is_header(line):
            break
        collected.append(line)
        if len(collected) >= max_lines:
            break
    return "\n".join(collected).strip()


def _build_patient_summary_source(emr_text: str) -> str:
    t = (emr_text or "").strip()
    if not t:
        return ""

    head = t[:2000].strip()
    tail = t[-2000:].strip() if len(t) > 2000 else ""

    parts: List[str] = []

    def add(label: str, block: str) -> None:
        if not block:
            return
        parts.append(f"[{label}]\n{block}".strip())

    add("HEADER", head)
    add("FAMILY HISTORY", _extract_section_block_last(t, ["Family Hx", "Family History"], max_lines=14))
    add("SOCIAL HISTORY", _extract_section_block_last(t, ["Social History", "Social Hx"], max_lines=25))
    add("HEALTH PROFILE", _extract_section_block_last(t, ["Health Profile", "Problem List", "Diagnoses"], max_lines=20))
    add("ACTIVE CONDITIONS", _extract_section_block_last(t, ["Active & Past Medical Conditions", "Active Medical Conditions"], max_lines=80))
    add("CLINICAL NOTES", _extract_section_block_last(t, ["Clinical Notes"], max_lines=60))
    add("PATIENT NOTES", _extract_section_block_last(t, ["Patient Notes"], max_lines=200))
    if tail:
        add("TAIL", tail)

    merged = "\n\n".join(parts).strip()
    return _clip_text(merged, max_chars=PATIENT_SUMMARY_MAX_CHARS)


def _summary_has_details(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    text_fields = [
        "family_members",
        "occupation",
        "hobbies_and_interests",
        "recent_social_life",
    ]
    for k in text_fields:
        if str(data.get(k) or "").strip():
            return True
    array_fields = [
        "recent_visits",
        "significant_life_events",
        "diagnoses",
    ]
    for k in array_fields:
        v = data.get(k)
        if isinstance(v, list) and any(str(x or "").strip() for x in v):
            return True
    return False


def _run_patient_summary_llm(prompt: str, model: str) -> Dict[str, Any]:
    if not model:
        return {}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only. Use EMR only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _build_patient_summary_prompt(emr_text: str, demographics: Dict[str, Any]) -> str:
    demo_block = "\n".join([
        f"Patient Name (source of truth): {demographics.get('patient_name','')}",
        f"PHN (source of truth): {demographics.get('phn','')}",
        f"Age (source of truth): {demographics.get('age_display','')}",
        f"Gender (source of truth): {demographics.get('gender','')}",
    ]).strip()

    return f"""
You are a Canadian (Alberta) primary care assistant.
Use ONLY the EMR text below. Do not invent details.

OUTPUT FORMAT:
Return STRICT JSON only (no markdown, no extra text).

JSON SCHEMA:
{{
  "patient_name": "",
  "phn": "",
  "age_display": "",
  "gender": "",
  "family_members": "",
  "occupation": "",
  "hobbies_and_interests": "",
  "recent_social_life": "",
  "recent_visits": ["..."],
  "significant_life_events": ["..."],
  "diagnoses": ["..."]
}}

RULES:
- Use the source-of-truth demographics if provided.
- Age must be shown as "X years" or "X months" if under 2 years.
- Gender must be one of: "Male", "Female", "Non-binary" (leave blank if unknown). If source-of-truth is blank, you may use explicit gender stated in EMR text.
- Prefer details explicitly stated in Social History, Family History, and recent note content (e.g., Patient Notes/Clinical Notes).
- Family members: include names if present; otherwise "".
- Recent social life: brief phrase if present; otherwise "".
- Recent visits: include only if the most recent visit is within the last 12 months. One short sentence per visit; most recent first.
- Significant life events: most recent first; return [] if none.
- Diagnoses: one line each, most relevant/recent first; add date if available. Return [] if none.

{demo_block}

EMR TEXT:
{emr_text}
""".strip()


def generate_patient_summary(context: SessionContext) -> Dict[str, Any]:
    emr = (getattr(context.clinical_background, "emr_dump", None) or "").strip()
    if not emr:
        return {}

    name, phn, dob, age, sex = extract_demographics_from_text(emr)
    age_display = _age_display_from_demographics(dob or "", age)
    gender = _gender_display(sex or "")

    demo = {
        "patient_name": name or "",
        "phn": phn or "",
        "age_display": age_display or "",
        "gender": gender or "",
    }

    emr_focus = _build_patient_summary_source(emr)
    prompt = _build_patient_summary_prompt(emr_focus, demo)

    data = _run_patient_summary_llm(prompt, PATIENT_SUMMARY_MODEL)
    if (not _summary_has_details(data)) and PATIENT_SUMMARY_FALLBACK_MODEL != PATIENT_SUMMARY_MODEL:
        fallback = _run_patient_summary_llm(prompt, PATIENT_SUMMARY_FALLBACK_MODEL)
        if _summary_has_details(fallback):
            data = fallback

    # Ensure required keys and inject deterministic demographics if missing
    for k, v in demo.items():
        if not (data.get(k) or "").strip():
            data[k] = v

    data.setdefault("family_members", "")
    data.setdefault("occupation", "")
    data.setdefault("hobbies_and_interests", "")
    data.setdefault("recent_social_life", "")
    data.setdefault("recent_visits", [])
    data.setdefault("significant_life_events", [])
    data.setdefault("diagnoses", [])

    return data


# =============================================================================
# BILLING (DAILY LIST) - SERVER-SIDE STATE + AUTOBILL
# =============================================================================

BILLING_DATA_DIR = os.getenv("BILLING_DATA_DIR", "./data/billing")
_BILLING_LOCK = threading.Lock()


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _billing_date_key(d: Optional[date] = None) -> str:
    dd = d or date.today()
    return dd.isoformat()


def _billing_state_path(date_key: str) -> str:
    _ensure_dir(BILLING_DATA_DIR)
    safe = re.sub(r"[^0-9\-]", "", date_key or "")
    if not safe:
        safe = _billing_date_key()
    return os.path.join(BILLING_DATA_DIR, f"billing_{safe}.json")


@dataclass
class BillingEntry:
    entry_id: str
    created_at_utc: str
    # Display lines (exactly the 3-line block the frontend renders/edits)
    line1_patient: str
    line2_icd9: str
    line3_billing: str
    # Optional structured data for future features
    meta: Dict[str, Any]


@dataclass
class BillingDayState:
    date_key: str
    physician: str
    billing_model: str  # "FFS" or "PCPCM"
    entries: List[BillingEntry]
    updated_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date_key": self.date_key,
            "physician": self.physician or "",
            "billing_model": self.billing_model or "FFS",
            "entries": [asdict(e) for e in (self.entries or [])],
            "updated_at_utc": self.updated_at_utc,
            "total_patient_count": len(self.entries or []),
        }


def _default_day_state(date_key: Optional[str] = None) -> BillingDayState:
    dk = date_key or _billing_date_key()
    return BillingDayState(
        date_key=dk,
        physician="",
        billing_model="FFS",
        entries=[],
        updated_at_utc=_now_utc().isoformat(),
    )


def _load_day_state(date_key: Optional[str] = None) -> BillingDayState:
    dk = date_key or _billing_date_key()
    path = _billing_state_path(dk)
    try:
        if not os.path.exists(path):
            return _default_day_state(dk)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return _default_day_state(dk)

        entries_in = obj.get("entries") or []
        entries: List[BillingEntry] = []
        if isinstance(entries_in, list):
            for it in entries_in:
                if not isinstance(it, dict):
                    continue
                entries.append(
                    BillingEntry(
                        entry_id=str(it.get("entry_id") or uuid.uuid4().hex),
                        created_at_utc=str(it.get("created_at_utc") or _now_utc().isoformat()),
                        line1_patient=str(it.get("line1_patient") or ""),
                        line2_icd9=str(it.get("line2_icd9") or ""),
                        line3_billing=str(it.get("line3_billing") or ""),
                        meta=it.get("meta") if isinstance(it.get("meta"), dict) else {},
                    )
                )

        return BillingDayState(
            date_key=str(obj.get("date_key") or dk),
            physician=str(obj.get("physician") or ""),
            billing_model=str(obj.get("billing_model") or "FFS"),
            entries=entries,
            updated_at_utc=str(obj.get("updated_at_utc") or _now_utc().isoformat()),
        )
    except Exception:
        return _default_day_state(dk)


def _persist_day_state(state: BillingDayState) -> None:
    dk = state.date_key or _billing_date_key()
    path = _billing_state_path(dk)
    try:
        state.updated_at_utc = _now_utc().isoformat()
        payload = state.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def billing_get_day_state(date_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Read the current day's billing list (server-side).
    """
    with _BILLING_LOCK:
        state = _load_day_state(date_key)
        return state.to_dict()


def billing_set_header(
    physician: str = "",
    billing_model: str = "FFS",
    date_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sets the "header" fields (physician + model) and persists to disk.
    """
    bm = (billing_model or "FFS").strip().upper()
    if bm not in ("FFS", "PCPCM"):
        bm = "FFS"

    with _BILLING_LOCK:
        state = _load_day_state(date_key)
        state.physician = (physician or "").strip()
        state.billing_model = bm
        _persist_day_state(state)
        return state.to_dict()


def _age_display_from_identifiers(ids: Dict[str, Any]) -> str:
    """
    Returns "55 years" or "9 months" (if < 24 months and DOB is available).
    Falls back to "X years" if only age-years is known.
    """
    age_years = ids.get("age", None)
    dob_s = (ids.get("dob") or "").strip()

    dob_dt = _parse_date_any(dob_s) if dob_s else None
    today = date.today()

    if dob_dt:
        m = _months_between(dob_dt, today)
        if m < 24:
            return f"{m} months"
        y = _age_from_dob(dob_dt)
        if y is not None:
            return f"{y} years"

    try:
        if age_years is not None:
            y = int(age_years)
            if y < 2 and dob_dt is not None:
                m = _months_between(dob_dt, today)
                return f"{m} months"
            return f"{y} years"
    except Exception:
        pass

    return ""


def _format_patient_line_from_context(context: SessionContext) -> Tuple[str, Dict[str, Any]]:
    """
    Line 1 spec:
      Name | PHN | age display
    Extract from chat EMR dump data (background).
    """
    ids = _extract_identifiers_from_background(context)
    name = (ids.get("name") or "").strip()
    phn = _clean_phn((ids.get("phn") or "").strip())
    age_disp = _age_display_from_identifiers(ids)

    parts = []
    parts.append(name if name else "[Patient]")
    parts.append(phn if len(phn) == 9 else "")
    parts.append(age_disp if age_disp else "")

    # Normalize: keep pipes but avoid trailing blanks
    line = " | ".join([p for p in parts if p != ""])
    if " | " not in line:
        # If only name is present, still show placeholders minimally
        line = (name or "[Patient]").strip()

    return line.strip(), ids


def _billing_reference_table_for_prompt() -> str:
    """
    Local (non-web) reference table that your billing prompt can reliably use.
    This matches your previously provided Alberta FFS reference list.
    """
    return """
Core (Alberta FFS) examples:
- 03.03A regular office visit; complexity modifiers CMGP01–CMGP10; G75GP if age ≥75 (20%); base $40.23.
- 03.04A comprehensive office visit (V70), CMXC30, 1/pt/345d, $110.64.
- 03.03CV virtual phone/video, CMGP01 only $19.54 add-on, total $59.77 (admin tasks excluded).
- 03.05JR physician-initiated results discussion, $20.00 (max 14/wk).
- 03.05JQ discuss acute psych concern with family (claimable with visit), $51.71.
- 03.07B repeat consultation $40.23 (+CMGP if applicable).
- 03.08A comprehensive consultation $131.40.

Procedures / add-ons examples:
- 13.99BA Pap ($30.17)
- 13.99BE pelvic/speculum+swabs ($30.17)
- 80.83B endometrial biopsy ($45.48)
- 12.23 vaginal foreign body removal ($86.82)
- 81.8 IUD insertion ($71.48)
- 11.71A IUD removal ($37.24)
- 93.91A hip injection/aspiration ($37.48)
- 93.91B other joint injection/aspiration ($19.98; max 2/day; 2nd @ 75%)
- 98.12J bursa/ganglion/tendon injection ($13.35)
- 98.03A I&D abscess/hematoma ($16.69)
- 98.12A/B excisional biopsy skin/face ($44.39/$56.93)
- 98.12C sebaceous cyst removal ($38.56; 2/3rd @ 75%)
- 98.22A/B laceration repair ($60.34/$63.69; LA included; not claimable with visit)
""".strip()


def _build_billing_suggestion_prompt(
    billing_model: str,
    background_text: str,
    transcript_text: str,
) -> str:
    """
    The LLM must:
    - Extract ICD-9 strictly from today's transcript (not background)
    - Suggest Alberta billing codes (maximize compliantly)
    - Respect billing model (FFS vs PCPCM)
    """
    bm = (billing_model or "FFS").strip().upper()
    if bm not in ("FFS", "PCPCM"):
        bm = "FFS"

    return f"""
You are assisting a licensed Alberta family physician with DAILY BILLING LINE SUGGESTIONS.

Jurisdiction: Alberta, Canada.

Billing model for this encounter: {bm}

STRICT RULES:
1) ICD-9: include ONLY diagnoses that were discussed/managed TODAY, based strictly on the VISIT TRANSCRIPT.
   - Do NOT include PMHx from the background unless it was actively addressed today.
   - If transcript is present, return at least one best-guess ICD-9 code.
2) Billing codes:
   - Suggest the most appropriate Alberta primary care physician billing code(s) for the encounter.
   - Use Alberta family practice billing conventions and avoid stacking multiple base visit codes.
   - Include only ONE primary visit/consult code (03.xx) per encounter.
   - Try to maximize billing compliantly.
   - Assume baseline non-face-to-face time is already spent: chart review 5 min + charting 5 min (=10 minutes).
   - If the transcript clearly implies extra time/complexity, reflect that with appropriate complexity/time modifiers where reasonable.
3) PCPCM:
   - Still produce a “shadow-billing style” code suggestion that mirrors the service delivered.
4) Output must be STRICT JSON ONLY, no markdown, no extra text.

REFERENCE TABLE (local excerpt; use as anchor when possible):
{_billing_reference_table_for_prompt()}

VISIT TRANSCRIPT (source of truth for today's problems):
{_clip_text(transcript_text or "", 5200) if (transcript_text or "").strip() else "None"}

CLINICAL BACKGROUND (secondary; do not mine diagnoses unless addressed today):
{_clip_text(background_text or "", 2600) if (background_text or "").strip() else "None"}

Return JSON with this schema:

{{
  "icd9": [{{"code":"401", "dx":"Hypertension"}}, ...],
  "billing": [
     {{"code":"03.03A"}},
     {{"code":"CMGP01"}},
     {{"code":"93.91A"}}
  ],
  "notes": "optional short note (1 line max) if uncertainty"
}}

Additional formatting rules:
- icd9[].code must be digits only (no periods).
- billing[].code must be the Alberta code string exactly (e.g., 03.03A, CMGP01).
- Do NOT include descriptors in the billing array.
""".strip()


def _format_icd9_line(icd9: List[Dict[str, str]]) -> str:
    if not icd9:
        return "ICD-9: "
    parts = []
    for it in icd9:
        if not isinstance(it, dict):
            continue
        code = str(it.get("code") or "").strip()
        dx = str(it.get("dx") or "").strip()
        if not code:
            continue
        if dx:
            parts.append(f"{code} ({dx})")
        else:
            parts.append(code)
    return "ICD-9: " + ", ".join(parts).strip()


def _format_billing_line(billing_items: List[Dict[str, str]]) -> str:
    """
    Line 3 spec:
      Billing: 03.03A + CMGP01 + 93.91A
    """
    if not billing_items:
        return "Billing: "

    codes: List[str] = []
    for it in billing_items:
        if not isinstance(it, dict):
            continue
        code = str(it.get("code") or "").strip()
        if not code:
            continue
        codes.append(code)

    base = "Billing: " + " + ".join(codes)
    return base.strip()


def _fallback_icd9_from_transcript(transcript_text: str) -> List[Dict[str, str]]:
    prompt = f"""
Extract ICD-9 codes from the VISIT TRANSCRIPT below. Use best-guess codes based strictly on what was discussed today.
Return STRICT JSON ONLY with schema:
{{"icd9":[{{"code":"401","dx":"Hypertension"}}]}}
- code must be digits only (no periods).
- If nothing is clearly discussed, return icd9=[].

VISIT TRANSCRIPT:
{_clip_text(transcript_text or "", 5200) if (transcript_text or "").strip() else "None"}
""".strip()
    try:
        resp = client.chat.completions.create(
            model=BILLING_MODEL,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only. Transcript-only ICD-9 extraction."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("icd9"), list):
            return obj.get("icd9") or []
    except Exception:
        return []
    return []


def billing_generate_entry_lines(
    context: SessionContext,
    billing_model: str = "FFS",
) -> Dict[str, Any]:
    """
    Generates the 3 lines for a single patient encounter:
      line1_patient
      line2_icd9
      line3_billing

    - Line 1 demographics strictly from EMR/background (chat EMR dump)
    - ICD-9 strictly from transcript (today)
    - Billing code suggestion based on transcript + model selection
    """
    # Line 1
    line1, ids = _format_patient_line_from_context(context)

    bg = (getattr(context.clinical_background, "emr_dump", None) or "").strip()
    tx = (getattr(context.transcript, "raw_text", None) or "").strip()

    # If transcript is missing, still return the demographic line and blanks for ICD/Billing
    if not tx:
        return {
            "line1_patient": line1,
            "line2_icd9": "ICD-9: ",
            "line3_billing": "Billing: ",
            "structured": {"icd9": [], "billing": [], "notes": "No transcript available yet."},
            "identifiers": ids,
        }

    prompt = _build_billing_suggestion_prompt(
        billing_model=billing_model,
        background_text=bg,
        transcript_text=tx,
    )

    # Call model, strict JSON if possible
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=BILLING_MODEL,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only. Be conservative with inferences; transcript-first."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        resp = client.chat.completions.create(
            model=BILLING_MODEL,
            messages=[
                {"role": "system", "content": "Return STRICT JSON only. Be conservative with inferences; transcript-first."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()

    parsed: Dict[str, Any] = {"icd9": [], "billing": [], "notes": ""}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        parsed = {"icd9": [], "billing": [], "notes": "Model returned non-JSON output."}

    icd9 = parsed.get("icd9") if isinstance(parsed.get("icd9"), list) else []
    billing_items = parsed.get("billing") if isinstance(parsed.get("billing"), list) else []
    if tx and not icd9:
        icd9 = _fallback_icd9_from_transcript(tx)
    line2 = _format_icd9_line(icd9)
    line3 = _format_billing_line(billing_items)

    return {
        "line1_patient": line1,
        "line2_icd9": line2,
        "line3_billing": line3,
        "structured": {"icd9": icd9, "billing": billing_items, "notes": str(parsed.get("notes") or "").strip()},
        "identifiers": ids,
        "meta": {
            "model": BILLING_MODEL,
            "generated_at_utc": _now_utc().isoformat(),
            "billing_model": (billing_model or "FFS").strip().upper(),
        },
    }


def billing_autobill_append(
    context: SessionContext,
    physician: str = "",
    billing_model: str = "FFS",
    date_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Called when frontend clicks "Bill".
    - Generates the 3-line entry
    - Appends to today's list
    - Persists to disk
    - Returns updated day state + new entry id
    """
    bm = (billing_model or "FFS").strip().upper()
    if bm not in ("FFS", "PCPCM"):
        bm = "FFS"

    entry_payload = billing_generate_entry_lines(context=context, billing_model=bm)

    with _BILLING_LOCK:
        state = _load_day_state(date_key)
        # Keep header fields in sync if provided
        if (physician or "").strip():
            state.physician = (physician or "").strip()
        state.billing_model = bm

        e = BillingEntry(
            entry_id=uuid.uuid4().hex,
            created_at_utc=_now_utc().isoformat(),
            line1_patient=str(entry_payload.get("line1_patient") or "").strip(),
            line2_icd9=str(entry_payload.get("line2_icd9") or "").strip(),
            line3_billing=str(entry_payload.get("line3_billing") or "").strip(),
            meta={
                "structured": entry_payload.get("structured") if isinstance(entry_payload.get("structured"), dict) else {},
                "identifiers": entry_payload.get("identifiers") if isinstance(entry_payload.get("identifiers"), dict) else {},
                "generation_meta": entry_payload.get("meta") if isinstance(entry_payload.get("meta"), dict) else {},
            },
        )
        state.entries.append(e)
        _persist_day_state(state)

        out = state.to_dict()
        out["new_entry_id"] = e.entry_id
        return out


def billing_save_day_state(
    payload: Dict[str, Any],
    date_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Called when frontend clicks "Save" after editing.

    Expected payload shape (frontend can send exactly what it displays):
    {
      "physician": "...",
      "billing_model": "FFS"|"PCPCM",
      "entries": [
        {"entry_id":"...", "line1_patient":"...", "line2_icd9":"...", "line3_billing":"...", "meta": {...}}
      ]
    }
    """
    dk = date_key or _billing_date_key()
    physician = str(payload.get("physician") or "").strip()
    bm = str(payload.get("billing_model") or "FFS").strip().upper()
    if bm not in ("FFS", "PCPCM"):
        bm = "FFS"

    entries_in = payload.get("entries") or []
    new_entries: List[BillingEntry] = []
    if isinstance(entries_in, list):
        for it in entries_in:
            if not isinstance(it, dict):
                continue
            new_entries.append(
                BillingEntry(
                    entry_id=str(it.get("entry_id") or uuid.uuid4().hex),
                    created_at_utc=str(it.get("created_at_utc") or _now_utc().isoformat()),
                    line1_patient=str(it.get("line1_patient") or ""),
                    line2_icd9=str(it.get("line2_icd9") or ""),
                    line3_billing=str(it.get("line3_billing") or ""),
                    meta=it.get("meta") if isinstance(it.get("meta"), dict) else {},
                )
            )

    with _BILLING_LOCK:
        state = _load_day_state(dk)
        state.date_key = dk
        state.physician = physician
        state.billing_model = bm
        state.entries = new_entries
        _persist_day_state(state)
        return state.to_dict()


def _strip_icd9_descriptions(line2_icd9: str) -> str:
    """
    "ICD-9: 401 (Hypertension), 250 (Diabetes)" -> "ICD-9: 401, 250"
    """
    s = (line2_icd9 or "").strip()
    if not s:
        return "ICD-9: "
    if s.lower().startswith("icd-9"):
        s = re.sub(r"^icd-9\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        codes = re.findall(r"\b(\d{3,5})\b", s)
        return "ICD-9: " + ", ".join(codes)
    codes = re.findall(r"\b(\d{3,5})\b", s)
    return "ICD-9: " + ", ".join(codes)


def _strip_billing_descriptions(line3_billing: str) -> str:
    """
    "Billing: 03.03A + CMGP01 + 93.91A (Hip injection) ..." -> "Billing: 03.03A + CMGP01 + 93.91A"
    """
    s = (line3_billing or "").strip()
    if not s:
        return "Billing: "
    if s.lower().startswith("billing"):
        s = re.sub(r"^billing\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    codes = re.findall(r"\b(?:\d{2}\.\d{2}[A-Z]{0,2}|\d{2}\.\d{2}[A-Z]|\d{2}\.\d{2}[A-Z]{2}|[A-Z]{2,6}\d{1,2}[A-Z]{0,2})\b", s)
    out: List[str] = []
    for c in codes:
        if c not in out:
            out.append(c)
    return "Billing: " + " + ".join(out)


def billing_get_print_payload_and_clear(date_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Called when frontend clicks "Print".
    Returns a payload suitable to render/print, with:
      - ICD-9 descriptions stripped (codes only)
      - Billing descriptions stripped (codes only)
    Then clears server-side state for that day (memory + disk).
    """
    dk = date_key or _billing_date_key()
    path = _billing_state_path(dk)

    with _BILLING_LOCK:
        state = _load_day_state(dk)

        printable_entries = []
        for e in state.entries:
            printable_entries.append({
                "entry_id": e.entry_id,
                "line1_patient": e.line1_patient,
                "line2_icd9": _strip_icd9_descriptions(e.line2_icd9),
                "line3_billing": _strip_billing_descriptions(e.line3_billing),
            })

        payload = {
            "date_key": state.date_key,
            "physician": state.physician,
            "billing_model": state.billing_model,
            "total_patient_count": len(state.entries),
            "entries": printable_entries,
            "generated_at_utc": _now_utc().isoformat(),
        }

        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

        return payload
