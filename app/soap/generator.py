from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .prompts import (
    SOAP_COMPACT_SYSTEM,
    SOAP_COMPACT_USER,
    SOAP_DRAFT_SYSTEM,
    SOAP_DRAFT_USER,
    SOAP_FINAL_SYSTEM,
    SOAP_FINAL_USER,
)
from .renderer import render_soap
from .schema import SoapStructured, soap_json_schema

logger = logging.getLogger("centaurweb.soap")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SOAP_DRAFT_MODEL = os.getenv("SOAP_DRAFT_MODEL", "gpt-5.2-pro")
SOAP_FINAL_MODEL = os.getenv("SOAP_FINAL_MODEL", "gpt-5.2-chat-latest")
SOAP_REASONING_EFFORT = os.getenv("SOAP_REASONING_EFFORT", "medium")
SOAP_VERBOSITY = os.getenv("SOAP_VERBOSITY", "high")
SOAP_MAX_OUTPUT_TOKENS = int(os.getenv("SOAP_MAX_OUTPUT_TOKENS", "2500"))
SOAP_DEBUG_LOG_PROMPTS = os.getenv("SOAP_DEBUG_LOG_PROMPTS", "0").strip() == "1"
SOAP_SINGLE_PASS = os.getenv("SOAP_SINGLE_PASS", "1").strip() == "1"
SOAP_SINGLE_PASS_MODEL = os.getenv("SOAP_SINGLE_PASS_MODEL", "gpt-4o-mini")
SOAP_SINGLE_PASS_FALLBACK_MODEL = os.getenv("SOAP_SINGLE_PASS_FALLBACK_MODEL", "gpt-4.1-mini")
SOAP_SINGLE_PASS_EFFORT = os.getenv("SOAP_SINGLE_PASS_EFFORT", "low")
SOAP_SINGLE_PASS_VERBOSITY = os.getenv("SOAP_SINGLE_PASS_VERBOSITY", "medium")

SOAP_MAX_TRANSCRIPT_CHARS = int(os.getenv("SOAP_MAX_TRANSCRIPT_CHARS", "24000"))
SOAP_COMPACT_CHUNK_CHARS = int(os.getenv("SOAP_COMPACT_CHUNK_CHARS", "3000"))
SOAP_HTTP_TIMEOUT = int(os.getenv("SOAP_HTTP_TIMEOUT", "60"))


@dataclass
class SoapGenerationResult:
    text: str
    structured: Dict[str, Any]
    draft_text: str
    transcript_hash: str
    transcript_chars: int
    compaction_used: bool


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _responses_available() -> bool:
    return hasattr(client, "responses")


def _responses_create(payload: Dict[str, Any]) -> Any:
    if _responses_available():
        return client.responses.create(**payload)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for SOAP generation.")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=SOAP_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_output_text(resp: Any) -> str:
    if hasattr(resp, "output_text"):
        return getattr(resp, "output_text") or ""
    if isinstance(resp, dict):
        if resp.get("output_text"):
            return resp.get("output_text") or ""
        output = resp.get("output") or []
        for item in output:
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text":
                    return content.get("text") or ""
    return ""


def _usage_tokens(resp: Any) -> Tuple[Optional[int], Optional[int]]:
    usage = None
    if hasattr(resp, "usage"):
        usage = getattr(resp, "usage")
    elif isinstance(resp, dict):
        usage = resp.get("usage")
    if not usage:
        return None, None
    if isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens") or usage.get("completion_tokens")
    return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)


def _log_event(
    stage: str,
    model: str,
    transcript_hash: str,
    transcript_chars: int,
    ok: bool,
    input_tokens=None,
    output_tokens=None,
    effort: Optional[str] = None,
    verbosity: Optional[str] = None,
):
    logger.info(
        "soap.%s model=%s effort=%s verbosity=%s transcript_hash=%s transcript_chars=%s ok=%s input_tokens=%s output_tokens=%s",
        stage,
        model,
        effort,
        verbosity,
        transcript_hash[:12],
        transcript_chars,
        ok,
        input_tokens,
        output_tokens,
    )


def _call_response(payload: Dict[str, Any], stage: str, model: str, transcript_hash: str, transcript_chars: int) -> str:
    try:
        resp = _responses_create(payload)
        text = _extract_output_text(resp)
        inp, out = _usage_tokens(resp)
        _log_event(
            stage,
            model,
            transcript_hash,
            transcript_chars,
            True,
            inp,
            out,
            effort=(payload.get("reasoning") or {}).get("effort"),
            verbosity=(payload.get("text") or {}).get("verbosity"),
        )
        return text or ""
    except Exception as e:
        msg = str(e).lower()
        # Retry removing unsupported temperature
        if "temperature" in msg:
            payload.pop("temperature", None)
        # Retry removing reasoning/verbosity when unsupported
        if "reasoning" in msg:
            payload.pop("reasoning", None)
        if "verbosity" in msg:
            text_cfg = payload.get("text")
            if isinstance(text_cfg, dict):
                text_cfg.pop("verbosity", None)
        if "temperature" in msg or "reasoning" in msg or "verbosity" in msg:
            resp = _responses_create(payload)
            text = _extract_output_text(resp)
            inp, out = _usage_tokens(resp)
            _log_event(
                stage,
                model,
                transcript_hash,
                transcript_chars,
                True,
                inp,
                out,
                effort=(payload.get("reasoning") or {}).get("effort"),
                verbosity=(payload.get("text") or {}).get("verbosity"),
            )
            return text or ""
        _log_event(
            stage,
            model,
            transcript_hash,
            transcript_chars,
            False,
            None,
            None,
            effort=(payload.get("reasoning") or {}).get("effort"),
            verbosity=(payload.get("text") or {}).get("verbosity"),
        )
        raise


def _compact_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "fact": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["fact", "quote"],
                },
            }
        },
        "required": ["facts"],
    }


def _extract_facts_from_chunk(chunk: str, transcript_hash: str) -> List[str]:
    if not chunk:
        return []
    payload = {
        "model": SOAP_FINAL_MODEL,
        "input": [
            {"role": "system", "content": SOAP_COMPACT_SYSTEM},
            {"role": "user", "content": SOAP_COMPACT_USER.format(chunk=chunk)},
        ],
        "temperature": 0.2,
        "max_output_tokens": 800,
        "reasoning": {"effort": "low"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "soap_compact",
                "schema": _compact_schema(),
                "strict": True,
            },
        },
    }
    raw = _call_response(payload, "compact", SOAP_FINAL_MODEL, transcript_hash, len(chunk))
    try:
        data = json.loads(raw)
    except Exception:
        return []
    facts = data.get("facts") or []
    out: List[str] = []
    for item in facts:
        fact = (item.get("fact") or "").strip()
        quote = (item.get("quote") or "").strip()
        if not fact:
            continue
        if quote:
            out.append(f"{fact} (\"{quote}\")")
        else:
            out.append(fact)
    return out


def _compact_transcript(transcript: str) -> str:
    transcript_hash = _sha256(transcript)
    chunks = _chunk_text(transcript, SOAP_COMPACT_CHUNK_CHARS)
    facts: List[str] = []
    for chunk in chunks:
        facts.extend(_extract_facts_from_chunk(chunk, transcript_hash))
    # Deduplicate
    seen = set()
    deduped: List[str] = []
    for fact in facts:
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return "\n".join(deduped).strip()


def _prepare_packet(packet: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    transcript = (packet.get("transcript", {}) or {}).get("text", "") or ""
    if len(transcript) <= SOAP_MAX_TRANSCRIPT_CHARS:
        return packet, False
    compacted = _compact_transcript(transcript)
    packet_copy = json.loads(json.dumps(packet))
    packet_copy["transcript"]["text"] = compacted or ""
    return packet_copy, True


def generate_soap_note(packet: Dict[str, Any]) -> SoapGenerationResult:
    transcript = (packet.get("transcript", {}) or {}).get("text", "") or ""
    transcript_hash = _sha256(transcript)
    transcript_chars = len(transcript)

    packet_for_prompt, compacted = _prepare_packet(packet)
    if compacted:
        logger.info("soap.compact transcript_hash=%s transcript_chars=%s", transcript_hash[:12], transcript_chars)
    transcript_for_prompt = (packet_for_prompt.get("transcript", {}) or {}).get("text", "") or ""
    if compacted and transcript_for_prompt:
        transcript_for_prompt = (
            "Transcript exceeded size limits; using compacted facts with quotes below.\n"
            + transcript_for_prompt
        )

    if SOAP_DEBUG_LOG_PROMPTS:
        logger.warning("SOAP_DEBUG_LOG_PROMPTS=1; logging prompts to stdout (PHI risk).")

    context_blob = json.dumps(packet_for_prompt, ensure_ascii=False)
    draft_user = SOAP_DRAFT_USER.format(
        transcript=transcript_for_prompt or "[empty]",
        context=context_blob,
    )

    draft_text = ""
    if SOAP_SINGLE_PASS:
        final_user = SOAP_FINAL_USER.format(
            transcript=transcript_for_prompt or "[empty]",
            packet=context_blob,
            draft="[no draft]",
        )
        payload = {
            "model": SOAP_SINGLE_PASS_MODEL,
            "input": [
                {"role": "system", "content": SOAP_FINAL_SYSTEM},
                {"role": "user", "content": final_user},
            ],
            "temperature": 0.2,
            "max_output_tokens": SOAP_MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": SOAP_SINGLE_PASS_EFFORT},
            "text": {
                "verbosity": SOAP_SINGLE_PASS_VERBOSITY,
                "format": {
                    "type": "json_schema",
                    "name": "soap_note",
                    "schema": soap_json_schema(),
                    "strict": True,
                },
            },
        }
        if SOAP_DEBUG_LOG_PROMPTS:
            logger.info("SOAP single-pass prompt: %s", final_user)
        try:
            raw_json = _call_response(payload, "single", SOAP_SINGLE_PASS_MODEL, transcript_hash, transcript_chars)
        except Exception as e:
            logger.warning("SOAP single-pass failed; retrying with fallback model: %s", e)
            payload["model"] = SOAP_SINGLE_PASS_FALLBACK_MODEL
            raw_json = _call_response(payload, "single_fallback", SOAP_SINGLE_PASS_FALLBACK_MODEL, transcript_hash, transcript_chars)
    else:
        try:
            payload = {
                "model": SOAP_DRAFT_MODEL,
                "input": [
                    {"role": "system", "content": SOAP_DRAFT_SYSTEM},
                    {"role": "user", "content": draft_user},
                ],
                "temperature": 0.2,
                "max_output_tokens": SOAP_MAX_OUTPUT_TOKENS,
                "reasoning": {"effort": SOAP_REASONING_EFFORT},
                "text": {"verbosity": SOAP_VERBOSITY},
            }
            if SOAP_DEBUG_LOG_PROMPTS:
                logger.info("SOAP draft prompt: %s", draft_user)
            draft_text = _call_response(payload, "draft", SOAP_DRAFT_MODEL, transcript_hash, transcript_chars)
        except Exception:
            draft_text = ""

        final_user = SOAP_FINAL_USER.format(
            transcript=transcript_for_prompt or "[empty]",
            packet=context_blob,
            draft=draft_text or "[no draft]",
        )

        payload = {
            "model": SOAP_FINAL_MODEL,
            "input": [
                {"role": "system", "content": SOAP_FINAL_SYSTEM},
                {"role": "user", "content": final_user},
            ],
            "temperature": 0.2,
            "max_output_tokens": SOAP_MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": SOAP_REASONING_EFFORT},
            "text": {
                "verbosity": SOAP_VERBOSITY,
                "format": {
                    "type": "json_schema",
                    "name": "soap_note",
                    "schema": soap_json_schema(),
                    "strict": True,
                },
            },
        }
        if SOAP_DEBUG_LOG_PROMPTS:
            logger.info("SOAP final prompt: %s", final_user)
        raw_json = _call_response(payload, "final", SOAP_FINAL_MODEL, transcript_hash, transcript_chars)

    try:
        parsed = json.loads(raw_json)
    except Exception:
        parsed = {}

    try:
        if hasattr(SoapStructured, "model_validate"):
            structured = SoapStructured.model_validate(parsed)
        else:
            structured = SoapStructured.parse_obj(parsed)
        structured_dict = structured.model_dump() if hasattr(structured, "model_dump") else structured.dict()
    except Exception:
        structured_dict = {
            "issues": [],
            "subjective": [],
            "safety_red_flags": ["none"],
            "social_hx": [],
            "objective": [],
            "assessment": [],
            "plan": [],
        }
        if hasattr(SoapStructured, "model_validate"):
            structured = SoapStructured.model_validate(structured_dict)
        else:
            structured = SoapStructured.parse_obj(structured_dict)

    final_text = render_soap(structured)

    return SoapGenerationResult(
        text=final_text,
        structured=structured_dict,
        draft_text=draft_text,
        transcript_hash=transcript_hash,
        transcript_chars=transcript_chars,
        compaction_used=compacted,
    )
