import os
import re
import tempfile
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger("centaurweb.transcription")

# Prefer explicit key in env; OpenAI() will also read OPENAI_API_KEY automatically
client = OpenAI()

# Default to your chosen STT model; allow override via env
MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

# Conservative min size (prevents empty/partial container chunks)
MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "3000"))

# Optional: language hint (e.g., "en"). If unset, model auto-detects.
LANGUAGE_HINT = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip() or None

# If true, we will return English-only text (best effort) by translating non-English.
TRANSLATE_TO_EN = (os.getenv("TRANSCRIBE_TRANSLATE_TO_EN", "0").strip() == "1")

# If true, suppress obvious non-speech junk (repeat syllables, subtitle artifacts).
SUPPRESS_NOISE = (os.getenv("TRANSCRIBE_NOISE_FILTER", "1").strip() == "1")

# Commonly accepted extensions by OpenAI STT
_ALLOWED_EXTS = {".webm", ".mp4", ".m4a", ".wav", ".mp3", ".mpeg", ".mpga", ".ogg"}


def _safe_ext(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _ALLOWED_EXTS else ".webm"


def _normalize_whitespace(text: str) -> str:
    # Keep it simple: normalize weird whitespace and trim
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = " ".join(t.split())  # collapses all runs of whitespace to single spaces
    return t.strip()


def _looks_english(text: str) -> bool:
    """
    Very cheap heuristic to avoid translating already-English text.
    """
    if not text:
        return True
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return ascii_ratio > 0.92


def _translate_to_english(text: str) -> str:
    """
    Best-effort translation to English using a lightweight chat call.
    Only used if TRANSLATE_TO_EN=1 and we suspect non-English text.
    """
    if not text or _looks_english(text):
        return text

    try:
        resp = client.chat.completions.create(
            model=os.getenv("TRANSCRIBE_TRANSLATE_MODEL", "gpt-4.1-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Translate to English. Return ONLY the translation; no quotes or commentary.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip()
        return _normalize_whitespace(out) or text
    except Exception as e:
        logger.warning(f"Translation failed; returning original text: {e}")
        return text


_BRACKETED_NOISE_RE = re.compile(r"\[(music|applause|laughter|noise|silence|background|inaudible)\]", re.I)
_NOISE_PHRASES = (
    "amara.org",
    "sottotitoli creati",
    "subtitles created",
    "subtitles by",
)


def _is_repeated_unit(token: str, max_unit: int = 3, min_repeats: int = 5) -> bool:
    clean = re.sub(r"[^A-Za-z]", "", token or "")
    if len(clean) < max_unit * min_repeats:
        return False
    if len(set(clean)) <= 2:
        return True
    for unit_len in range(1, max_unit + 1):
        if len(clean) % unit_len != 0:
            continue
        unit = clean[:unit_len]
        repeats = len(clean) // unit_len
        if repeats >= min_repeats and unit * repeats == clean:
            return True
    return False


def _suppress_nonsense(text: str) -> str:
    if not text:
        return text
    t = _BRACKETED_NOISE_RE.sub("", text)
    for phrase in _NOISE_PHRASES:
        t = re.sub(re.escape(phrase), "", t, flags=re.I)
    tokens = t.split()
    kept = []
    for tok in tokens:
        if _is_repeated_unit(tok):
            continue
        kept.append(tok)
    t = " ".join(kept).strip()
    return _normalize_whitespace(t)


def _is_repetitive_phrase(tokens: list[str], max_len: int = 4) -> bool:
    if len(tokens) < 8:
        return False
    max_len = max(2, min(max_len, len(tokens) // 3))
    for n in range(2, max_len + 1):
        phrases = {}
        for i in range(0, len(tokens) - n + 1):
            key = " ".join(tokens[i:i + n])
            phrases[key] = phrases.get(key, 0) + 1
        if not phrases:
            continue
        phrase, count = max(phrases.items(), key=lambda kv: kv[1])
        if count >= 3:
            covered = count * n
            if covered / max(1, len(tokens)) >= 0.6:
                return True
    return False


def _is_repetitive_sentence(text: str) -> bool:
    parts = [p.strip().lower() for p in re.split(r"[.!?]+", text) if p.strip()]
    if len(parts) < 3:
        return False
    counts = {}
    for p in parts:
        counts[p] = counts.get(p, 0) + 1
    sent, count = max(counts.items(), key=lambda kv: kv[1])
    if count >= 3 and len(sent.split()) <= 8:
        return True
    return False


def _should_drop_transcript(text: str) -> bool:
    if not text:
        return True
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if len(tokens) < 4:
        return False
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if len(tokens) >= 10 and unique_ratio < 0.35:
        return True
    if _is_repetitive_phrase(tokens):
        return True
    if _is_repetitive_sentence(text):
        return True
    return False


def transcribe_audio_bytes(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio bytes using OpenAI Speech-to-Text.

    Behavior:
      - Returns transcript in spoken language by default.
      - If TRANSCRIBE_TRANSLATE_TO_EN=1, returns English-only transcript (best effort).
      - Uses a safe temp file and always cleans up.
      - Adds optional language hint via TRANSCRIBE_LANGUAGE (e.g., 'en').

    Notes:
      - Do not pass tiny blobs; you are already chunking in frontend.
      - Keep temperature=0 for determinism in clinical workflows.
    """
    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        return ""

    ext = _safe_ext(filename or "")
    temp_path: Optional[str] = None

    try:
        # Use delete=False so Windows and some sandbox FS allow re-open for reading
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        with open(temp_path, "rb") as af:
            kwargs = {
                "model": MODEL,
                "file": af,
                "response_format": "text",
                "temperature": 0,
            }
            # Language hint improves stability/latency when you know the default
            if LANGUAGE_HINT:
                kwargs["language"] = LANGUAGE_HINT

            tr = client.audio.transcriptions.create(**kwargs)

        # SDK may return either an object with .text or a plain string
        if isinstance(tr, str):
            text = tr
        else:
            text = getattr(tr, "text", "") or ""

        text = _normalize_whitespace(text)

        if not text or len(text) < 2:
            return ""

        if TRANSLATE_TO_EN:
            text = _translate_to_english(text)

        if SUPPRESS_NOISE:
            text = _suppress_nonsense(text)

        if _should_drop_transcript(text):
            return ""

        if not text:
            return ""

        return text

    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        # Surface as empty; api.py can decide whether to raise or tolerate
        return ""

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
