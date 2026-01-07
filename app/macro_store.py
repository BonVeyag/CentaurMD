from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "macros.json",
)
ENV_STORE_PATH = "CENTAUR_MACRO_STORE_PATH"

_LOCK = threading.Lock()

SYSTEM_MACRO_IDS = {
    "SOAP": "system:soap",
    "REFERRAL": "system:referral",
}

DEFAULT_SOAP_TEMPLATE = """You are a Canadian family medicine clinical documentation assistant.

Jurisdiction: Alberta, Canada.

FORMATTING RULES:
- Bold section titles
- No bullets
- One blank line between sections
- Do NOT invent information

SECTIONS:
Issues
Subjective
Safety / red flags
Objective
Assessment
Plan

CLINICAL BACKGROUND:
{{EMR}}

VISIT TRANSCRIPT:
{{TRANSCRIPT}}

Generate the SOAP note now.
""".strip()

DEFAULT_REFERRAL_TEMPLATE = """Write a referral letter in Canadian/Alberta style.
Mandatory formatting:
- Start the letter body with: "Dear Colleague,"
- Use clear headings and short paragraphs.
- End the letter with exactly:
  Sincerely,
  (and DO NOT include any clinician name or credentials after Sincerely.)
Content requirements:
- Include: reason for referral, focused HPI, pertinent PMHx, meds/allergies if available, relevant exam/labs/imaging, what has been tried, specific question(s) for the consultant, urgency, and follow-up plan.
- If key details are missing, leave placeholders like [Consultant Name], [Clinic], [Fax], [Patient Name], [DOB], [PHN].
Extra instructions (if any):
{{EXTRA}}
Background/EMR (if any):
{{EMR}}
Live transcript:
{{TRANSCRIPT}}
""".strip()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_path() -> str:
    p = (os.getenv(ENV_STORE_PATH) or DEFAULT_STORE_PATH).strip()
    return p or DEFAULT_STORE_PATH


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _safe_json_load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _safe_json_write(path: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _clean_str(x: Any) -> str:
    return (str(x).strip() if x is not None else "")


@dataclass
class MacroEntry:
    id: str
    name: str
    content: str
    created_at_utc: str
    updated_at_utc: str
    locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_entries() -> List[MacroEntry]:
    path = _store_path()
    data = _safe_json_load(path)
    raw = data.get("macros", [])
    if not isinstance(raw, list):
        return []
    out: List[MacroEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        mid = _clean_str(item.get("id"))
        name = _clean_str(item.get("name"))
        content = _clean_str(item.get("content"))
        created_at_utc = _clean_str(item.get("created_at_utc"))
        updated_at_utc = _clean_str(item.get("updated_at_utc"))
        locked = bool(item.get("locked", False))
        if not mid or not name or not content:
            continue
        if not created_at_utc:
            created_at_utc = _utcnow_iso()
        if not updated_at_utc:
            updated_at_utc = created_at_utc
        out.append(
            MacroEntry(
                id=mid,
                name=name,
                content=content,
                created_at_utc=created_at_utc,
                updated_at_utc=updated_at_utc,
                locked=locked,
            )
        )
    return out


def _save_entries(entries: List[MacroEntry]) -> None:
    path = _store_path()
    payload = {"macros": [e.to_dict() for e in entries]}
    _safe_json_write(path, payload)


def _find_by_name(entries: List[MacroEntry], name: str) -> Optional[MacroEntry]:
    target = (name or "").strip().lower()
    if not target:
        return None
    for e in entries:
        if e.name.strip().lower() == target:
            return e
    return None


def _ensure_defaults(entries: List[MacroEntry]) -> tuple[List[MacroEntry], bool]:
    now = _utcnow_iso()
    changed = False

    soap_entry = _find_by_name(entries, "SOAP")
    if soap_entry:
        soap_changed = False
        if not soap_entry.locked:
            soap_entry.locked = True
            soap_changed = True
        if not soap_entry.content:
            soap_entry.content = DEFAULT_SOAP_TEMPLATE
            soap_changed = True
        if soap_changed:
            soap_entry.updated_at_utc = now
            changed = True
    else:
        entries.append(
            MacroEntry(
                id=SYSTEM_MACRO_IDS["SOAP"],
                name="SOAP",
                content=DEFAULT_SOAP_TEMPLATE,
                created_at_utc=now,
                updated_at_utc=now,
                locked=True,
            )
        )
        changed = True

    referral_entry = _find_by_name(entries, "Referral")
    if referral_entry:
        ref_changed = False
        if not referral_entry.locked:
            referral_entry.locked = True
            ref_changed = True
        if not referral_entry.content:
            referral_entry.content = DEFAULT_REFERRAL_TEMPLATE
            ref_changed = True
        if ref_changed:
            referral_entry.updated_at_utc = now
            changed = True
    else:
        entries.append(
            MacroEntry(
                id=SYSTEM_MACRO_IDS["REFERRAL"],
                name="Referral",
                content=DEFAULT_REFERRAL_TEMPLATE,
                created_at_utc=now,
                updated_at_utc=now,
                locked=True,
            )
        )
        changed = True

    return entries, changed


def list_macros() -> List[Dict[str, Any]]:
    with _LOCK:
        entries = _load_entries()
        entries, changed = _ensure_defaults(entries)
        if changed:
            _save_entries(entries)
    entries.sort(key=lambda m: m.updated_at_utc, reverse=True)
    return [e.to_dict() for e in entries]


def save_macro(macro_id: Optional[str], name: str, content: str) -> Dict[str, Any]:
    name = _clean_str(name)
    content = _clean_str(content)
    if not name or not content:
        raise ValueError("Macro name and content are required")

    with _LOCK:
        entries = _load_entries()
        entries, _ = _ensure_defaults(entries)
        now = _utcnow_iso()
        mid = _clean_str(macro_id) or str(uuid4())
        existing = None
        for idx, e in enumerate(entries):
            if e.id == mid:
                existing = (idx, e)
                break

        if existing:
            idx, e = existing
            updated = MacroEntry(
                id=e.id,
                name=name,
                content=content,
                created_at_utc=e.created_at_utc,
                updated_at_utc=now,
                locked=e.locked,
            )
            entries[idx] = updated
            _save_entries(entries)
            return updated.to_dict()

        created = MacroEntry(
            id=mid,
            name=name,
            content=content,
            created_at_utc=now,
            updated_at_utc=now,
            locked=False,
        )
        entries.append(created)
        _save_entries(entries)
        return created.to_dict()


def delete_macro(macro_id: str) -> bool:
    mid = _clean_str(macro_id)
    if not mid:
        return False
    with _LOCK:
        entries = _load_entries()
        entries, _ = _ensure_defaults(entries)
        for e in entries:
            if e.id == mid and e.locked:
                raise ValueError("Macro is locked and cannot be deleted")
        kept = [e for e in entries if e.id != mid]
        if len(kept) == len(entries):
            return False
        _save_entries(kept)
        return True
