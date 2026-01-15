import os
import re
import tempfile
import threading
import logging
from typing import Optional, Tuple, Dict, Any, Callable

from openai import OpenAI

logger = logging.getLogger("centaurweb.transcription")

# Prefer explicit key in env; OpenAI() will also read OPENAI_API_KEY automatically
client = OpenAI()

def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Transcription backend: local_whisper | openai
TRANSCRIBE_BACKEND = (os.getenv("TRANSCRIBE_BACKEND") or "local_whisper").strip().lower()

# Default to highest-accuracy STT model for OpenAI; allow override via env
MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe")

# Local Whisper settings
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base").strip()
WHISPER_DEVICE = (os.getenv("WHISPER_DEVICE") or "").strip()
WHISPER_FP16 = (os.getenv("WHISPER_FP16") or "0").strip() == "1"
WHISPER_CONDITION_ON_PREV = (os.getenv("WHISPER_CONDITION_ON_PREV") or "0").strip() == "1"
WHISPER_TEMPERATURE = _float_env("WHISPER_TEMPERATURE", 0.0)

# Conservative min size (prevents empty/partial container chunks)
MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "3000"))

# Optional: language hint (e.g., "en"). If unset, model auto-detects.
LANGUAGE_HINT = (os.getenv("TRANSCRIBE_LANGUAGE") or "").strip() or None

# STT response format (text or verbose_json).
RESPONSE_FORMAT = (os.getenv("TRANSCRIBE_RESPONSE_FORMAT") or "text").strip().lower()

TEMPERATURE = _float_env("TRANSCRIBE_TEMPERATURE", 0.0)

# Optional custom prompt for medical dictation (can be overridden per-call).
PROMPT_BASE = os.getenv(
    "TRANSCRIBE_PROMPT",
    "Clinical visit dictation in Canadian English. Preserve medical terms, drug names, labs, and units.",
).strip()
PROMPT_MAX_CHARS = int(os.getenv("TRANSCRIBE_PROMPT_MAX_CHARS", "1000"))
VOCAB_PATH = (os.getenv("TRANSCRIBE_VOCAB_PATH") or "").strip()
VOCAB_MAX_TERMS = int(os.getenv("TRANSCRIBE_VOCAB_MAX_TERMS", "80"))

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


_VOCAB_TERMS: Optional[list[str]] = None
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()
_FW_MODEL = None
_FW_LOCK = threading.Lock()


def _load_vocab_terms() -> list[str]:
    global _VOCAB_TERMS
    if _VOCAB_TERMS is not None:
        return _VOCAB_TERMS
    terms: list[str] = []
    if VOCAB_PATH and os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "," in line:
                        parts = [p.strip() for p in line.split(",") if p.strip()]
                        terms.extend(parts)
                    else:
                        terms.append(line)
        except Exception as e:
            logger.warning(f"Failed to load transcribe vocab file: {e}")
    # Deduplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    _VOCAB_TERMS = deduped
    return _VOCAB_TERMS


def _build_transcribe_prompt(
    prompt_override: Optional[str],
    extra_terms: Optional[list[str]] = None,
) -> Optional[str]:
    base = PROMPT_BASE if prompt_override is None else (prompt_override or "")
    terms = _load_vocab_terms()
    if extra_terms:
        terms = terms + extra_terms
    if terms:
        # De-dupe merged terms in order
        seen = set()
        merged: list[str] = []
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(term)
        vocab = ", ".join(merged[:VOCAB_MAX_TERMS])
        base = f"{base} Key terms: {vocab}".strip()
    base = (base or "").strip()
    if not base:
        return None
    if len(base) > PROMPT_MAX_CHARS:
        return base[:PROMPT_MAX_CHARS].rstrip()
    return base


def _load_faster_whisper_model():
    """
    Load faster-whisper base model (singleton).
    """
    global _FW_MODEL
    if _FW_MODEL is not None:
        return _FW_MODEL
    with _FW_LOCK:
        if _FW_MODEL is not None:
            return _FW_MODEL
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            logger.warning(f"faster-whisper import failed: {e}")
            return None
        device = WHISPER_DEVICE or "cpu"
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
        try:
            _FW_MODEL = WhisperModel(
                WHISPER_MODEL_NAME,
                device=device,
                compute_type=compute_type,
            )
        except Exception as e:
            logger.warning(f"faster-whisper load failed ({WHISPER_MODEL_NAME}): {e}")
            return None
    return _FW_MODEL


def _transcribe_with_faster_whisper(
    audio_path: str,
    prompt_text: Optional[str],
    language: Optional[str],
    *,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Tuple[str, Optional[str], Optional[float]]:
    """
    Transcribe with faster-whisper in TRANSCRIBE mode (no translation).
    Returns (text, language, language_probability).
    """
    model = _load_faster_whisper_model()
    if model is None:
        return "", None, None

    segments, info = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=prompt_text or None,
        condition_on_previous_text=False,
    )
    text_parts = []
    try:
        for seg in segments:
            if getattr(seg, "text", None):
                text_parts.append(seg.text)
    except Exception:
        pass
    text = " ".join(text_parts).strip()
    lang = getattr(info, "language", None)
    prob = getattr(info, "language_probability", None)
    return text, lang, prob


def _transcribe_with_openai(
    audio_file,
    prompt_text: Optional[str],
    language_hint: Optional[str],
) -> str:
    kwargs = {
        "model": MODEL,
        "file": audio_file,
        "response_format": RESPONSE_FORMAT,
        "temperature": TEMPERATURE,
    }
    if language_hint or LANGUAGE_HINT:
        kwargs["language"] = (language_hint or LANGUAGE_HINT)
    if prompt_text:
        kwargs["prompt"] = prompt_text
    tr = client.audio.transcriptions.create(**kwargs)
    if isinstance(tr, str):
        return tr
    text = getattr(tr, "text", "") or ""
    if not text and isinstance(tr, dict):
        text = tr.get("text") or tr.get("transcript") or ""
    return text


def _looks_english(text: str) -> bool:
    """
    Very cheap heuristic to avoid translating already-English text.
    """
    if not text:
        return True
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return ascii_ratio > 0.92

_SCRIPT_REGEX = {
    "ne": r"[\u0900-\u097F]",
    "hi": r"[\u0900-\u097F]",
    "mr": r"[\u0900-\u097F]",
    "zh": r"[\u4E00-\u9FFF]",
    "ja": r"[\u3040-\u30FF]",
    "ko": r"[\uAC00-\uD7AF]",
    "ar": r"[\u0600-\u06FF]",
    "fa": r"[\u0600-\u06FF]",
    "ur": r"[\u0600-\u06FF]",
    "ru": r"[\u0400-\u04FF]",
    "uk": r"[\u0400-\u04FF]",
    "bg": r"[\u0400-\u04FF]",
    "sr": r"[\u0400-\u04FF]",
    "el": r"[\u0370-\u03FF]",
    "he": r"[\u0590-\u05FF]",
    "th": r"[\u0E00-\u0E7F]",
}


def _count_chars(text: str, pattern: str) -> int:
    if not text or not pattern:
        return 0
    return len(re.findall(pattern, text))


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


def _has_repeated_ngram(tokens: list[str], n: int, min_count: int = 3, min_coverage: float = 0.35) -> bool:
    if len(tokens) < n * min_count:
        return False
    counts = {}
    for i in range(0, len(tokens) - n + 1):
        key = " ".join(tokens[i:i + n])
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return False
    _, count = max(counts.items(), key=lambda kv: kv[1])
    if count < min_count:
        return False
    coverage = (count * n) / max(1, len(tokens))
    return coverage >= min_coverage


def _should_drop_transcript(text: str) -> bool:
    if not text:
        return True
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if len(tokens) < 4:
        return False
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if len(tokens) >= 10 and unique_ratio < 0.35:
        return True
    if "i am a student" in text.lower() and text.lower().count("i am a student") >= 2 and unique_ratio < 0.6:
        return True
    if _has_repeated_ngram(tokens, 3, min_count=3, min_coverage=0.35):
        return True
    if _has_repeated_ngram(tokens, 4, min_count=3, min_coverage=0.35):
        return True
    if _is_repetitive_phrase(tokens):
        return True
    if _is_repetitive_sentence(text):
        return True
    return False


def transcribe_audio_bytes(
    audio_bytes: bytes,
    filename: str,
    *,
    prompt: Optional[str] = None,
    prompt_terms: Optional[list[str]] = None,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio bytes using local faster-whisper (base, int8).
    Returns dict: {text, language, language_prob}
    """
    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        return {"text": "", "language": None, "language_prob": None}

    ext = _safe_ext(filename or "")
    temp_path: Optional[str] = None

    try:
        # Use delete=False so Windows and some sandbox FS allow re-open for reading
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        prompt_text = _build_transcribe_prompt(prompt, prompt_terms)
        backend = TRANSCRIBE_BACKEND
        text = ""
        language_used: Optional[str] = language_hint or None
        language_prob: Optional[float] = None

        if backend in {"local_whisper", "whisper", "local"}:
            # First pass (auto-detect if no hint provided)
            lang_for_first_pass = language_used if language_used else None
            text, detected_lang, detected_prob = _transcribe_with_faster_whisper(
                temp_path,
                prompt_text,
                lang_for_first_pass,
                beam_size=5,
                vad_filter=True,
            )
            if not language_used and detected_lang:
                language_used = detected_lang
                language_prob = detected_prob
            if language_used and language_prob is None:
                language_prob = detected_prob

            # Script-guard retry for languages with a defined script
            if language_used and text:
                regex = _SCRIPT_REGEX.get(language_used)
                if regex:
                    count = _count_chars(text, regex)
                    if count < 3 and (language_prob or 0) >= 0.50:
                        retry_text, _, _ = _transcribe_with_faster_whisper(
                            temp_path,
                            prompt_text,
                            language_used,
                            beam_size=8,
                            vad_filter=False,
                        )
                        retry_count = _count_chars(retry_text, regex)
                        if retry_count > count:
                            text = retry_text

        # Fallback to OpenAI only if explicitly allowed and local returned empty
        if not text and os.getenv("TRANSCRIBE_FALLBACK_TO_OPENAI", "0").strip() == "1":
            with open(temp_path, "rb") as af:
                text = _transcribe_with_openai(af, prompt_text, language_hint)
                language_used = language_used or LANGUAGE_HINT or None

        text = _normalize_whitespace(text)

        if SUPPRESS_NOISE and text:
            text = _suppress_nonsense(text)

        if _should_drop_transcript(text):
            text = ""

        return {
            "text": text or "",
            "language": language_used,
            "language_prob": language_prob,
        }

    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        return {"text": "", "language": None, "language_prob": None}

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
