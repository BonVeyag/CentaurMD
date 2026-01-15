from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# =============================================================================
# Storage configuration
# =============================================================================

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "billing")
DEFAULT_DECISIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "billing_decisions")
ENV_DATA_DIR = "CENTAUR_BILLING_DATA_DIR"

# "1" => per-day file, else single rolling file
ENV_DAILY_FILES = "CENTAUR_BILLING_DAILY_FILES"

# default: "billing"
ENV_FILENAME_PREFIX = "CENTAUR_BILLING_FILENAME_PREFIX"
DEFAULT_PREFIX = "billing"

# local day boundary
ENV_TZ = "CENTAUR_TZ"
DEFAULT_TZ = "America/Edmonton"

_LOCK = threading.Lock()


# =============================================================================
# Time + helpers
# =============================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_tz_name() -> str:
    return (os.getenv(ENV_TZ) or DEFAULT_TZ).strip() or DEFAULT_TZ


def _local_today_key() -> str:
    """
    Local date key (America/Edmonton by default) to match clinic day boundaries.
    """
    tz_name = _get_tz_name()
    if ZoneInfo is None:
        return date.today().isoformat()
    try:
        return datetime.now(ZoneInfo(tz_name)).date().isoformat()
    except Exception:
        return date.today().isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _daily_files_enabled() -> bool:
    return (os.getenv(ENV_DAILY_FILES, "1").strip() == "1")


def _prefix() -> str:
    p = (os.getenv(ENV_FILENAME_PREFIX) or DEFAULT_PREFIX).strip()
    p = re.sub(r"[^a-zA-Z0-9_\-]+", "_", p) or DEFAULT_PREFIX
    return p


def _sanitize_date_key(k: str) -> str:
    safe = re.sub(r"[^0-9\-]", "", (k or "").strip())
    return safe or _local_today_key()


def _state_path(data_dir: str, date_key: Optional[str] = None) -> str:
    _ensure_dir(data_dir)
    dk = _sanitize_date_key(date_key or _local_today_key())
    if _daily_files_enabled():
        return os.path.join(data_dir, f"{_prefix()}_{dk}.json")
    return os.path.join(data_dir, f"{_prefix()}.json")


def _safe_json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_json_write(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def save_billing_decision(decision: Dict[str, Any], decisions_dir: Optional[str] = None) -> Optional[str]:
    """
    Persist a billing decision record (audit packet). Stores JSON per trace_id.
    By default, stores minimal data; caller controls the payload.
    """
    try:
        trace_id = (decision.get("trace_id") or str(uuid.uuid4())).strip()
        decisions_dir = decisions_dir or DEFAULT_DECISIONS_DIR
        _ensure_dir(decisions_dir)
        path = os.path.join(decisions_dir, f"{trace_id}.json")
        _safe_json_write(path, decision)
        return path
    except Exception:
        return None


def _normalize_model(model: str) -> str:
    """
    Accepts common variants:
      - "Fee for Service", "FFS"
      - "Primary Care Physician Compensation Model", "PCPCM"
    Returns: "FFS" or "PCPCM"
    """
    t = (model or "").strip().lower()
    if "pcpcm" in t:
        return "PCPCM"
    if "ffs" in t or "fee for service" in t:
        return "FFS"
    u = (model or "").strip().upper()
    if u in {"FFS", "PCPCM"}:
        return u
    return "FFS"


def _clean_str(x: Any) -> str:
    return (str(x).strip() if x is not None else "")


# =============================================================================
# Print stripping (matches your service.py behavior)
# =============================================================================

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
    "Billing: 03.03A + CMGP01 + 93.91A (Hip injection)" -> "Billing: 03.03A + CMGP01 + 93.91A"
    """
    s = (line3_billing or "").strip()
    if not s:
        return "Billing: "
    if s.lower().startswith("billing"):
        s = re.sub(r"^billing\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    # captures common Alberta-style codes + CMGP etc.
    codes = re.findall(
        r"\b(?:\d{2}\.\d{2}[A-Z]{0,2}|[A-Z]{2,6}\d{1,2}[A-Z]{0,2})\b",
        s
    )
    out: List[str] = []
    for c in codes:
        if c not in out:
            out.append(c)
    return "Billing: " + " + ".join(out)


# =============================================================================
# Data model
# =============================================================================

@dataclass
class BillingEntry:
    entry_id: str
    created_at_utc: str
    line1_patient: str
    line2_icd9: str
    line3_billing: str
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
            "billing_model": _normalize_model(self.billing_model or "FFS"),
            "entries": [asdict(e) for e in (self.entries or [])],
            "updated_at_utc": self.updated_at_utc,
            "total_patient_count": len(self.entries or []),
        }


# =============================================================================
# Store
# =============================================================================

class BillingStore:
    """
    Structured, per-day billing store aligned with your new frontend workflow:

    - Header: physician + billing_model (FFS/PCPCM) persisted per day
    - Entries: list of 3-line blocks (editable) with entry_id + meta
    - Save: replace entire day state from frontend payload
    - Print: returns stripped payload (codes only) and clears that day's file/state
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (data_dir or os.getenv(ENV_DATA_DIR) or DEFAULT_DATA_DIR).strip()
        _ensure_dir(self.data_dir)

    # -------------------------
    # Internal load/save
    # -------------------------

    def _default_state(self, date_key: Optional[str] = None) -> BillingDayState:
        dk = _sanitize_date_key(date_key or _local_today_key())
        return BillingDayState(
            date_key=dk,
            physician="",
            billing_model="FFS",
            entries=[],
            updated_at_utc=_utcnow_iso(),
        )

    def _load(self, date_key: Optional[str] = None) -> BillingDayState:
        dk = _sanitize_date_key(date_key or _local_today_key())
        path = _state_path(self.data_dir, dk)

        if not os.path.exists(path):
            return self._default_state(dk)

        try:
            obj = _safe_json_load(path)
            if not isinstance(obj, dict):
                return self._default_state(dk)

            entries_in = obj.get("entries") or []
            entries: List[BillingEntry] = []
            if isinstance(entries_in, list):
                for it in entries_in:
                    if not isinstance(it, dict):
                        continue
                    entries.append(
                        BillingEntry(
                            entry_id=_clean_str(it.get("entry_id") or uuid.uuid4().hex),
                            created_at_utc=_clean_str(it.get("created_at_utc") or _utcnow_iso()),
                            line1_patient=_clean_str(it.get("line1_patient")),
                            line2_icd9=_clean_str(it.get("line2_icd9")),
                            line3_billing=_clean_str(it.get("line3_billing")),
                            meta=it.get("meta") if isinstance(it.get("meta"), dict) else {},
                        )
                    )

            return BillingDayState(
                date_key=_sanitize_date_key(_clean_str(obj.get("date_key") or dk)),
                physician=_clean_str(obj.get("physician")),
                billing_model=_normalize_model(_clean_str(obj.get("billing_model") or "FFS")),
                entries=entries,
                updated_at_utc=_clean_str(obj.get("updated_at_utc") or _utcnow_iso()),
            )
        except Exception:
            return self._default_state(dk)

    def _persist(self, state: BillingDayState) -> None:
        state.updated_at_utc = _utcnow_iso()
        state.billing_model = _normalize_model(state.billing_model or "FFS")
        state.date_key = _sanitize_date_key(state.date_key or _local_today_key())
        path = _state_path(self.data_dir, state.date_key)
        _safe_json_write(path, state.to_dict())

    # -------------------------
    # Public API used by FastAPI routes / service layer
    # -------------------------

    def get_day_state(self, date_key: Optional[str] = None) -> Dict[str, Any]:
        with _LOCK:
            state = self._load(date_key)
            return state.to_dict()

    def set_header(
        self,
        physician: str = "",
        billing_model: str = "FFS",
        date_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        with _LOCK:
            state = self._load(date_key)
            state.physician = _clean_str(physician)
            state.billing_model = _normalize_model(billing_model)
            self._persist(state)
            return state.to_dict()

    def append_entry(
        self,
        line1_patient: str,
        line2_icd9: str,
        line3_billing: str,
        meta: Optional[Dict[str, Any]] = None,
        physician: str = "",
        billing_model: str = "",
        date_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Appends a new 3-line entry (what your frontend shows/edits).
        Optionally updates header (physician/model) if provided.
        """
        with _LOCK:
            state = self._load(date_key)

            if _clean_str(physician):
                state.physician = _clean_str(physician)
            if _clean_str(billing_model):
                state.billing_model = _normalize_model(billing_model)

            e = BillingEntry(
                entry_id=uuid.uuid4().hex,
                created_at_utc=_utcnow_iso(),
                line1_patient=_clean_str(line1_patient),
                line2_icd9=_clean_str(line2_icd9),
                line3_billing=_clean_str(line3_billing),
                meta=meta if isinstance(meta, dict) else {},
            )
            state.entries.append(e)
            self._persist(state)

            out = state.to_dict()
            out["new_entry_id"] = e.entry_id
            return out

    def save_day_state(
        self,
        payload: Dict[str, Any],
        date_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Replaces the server day's state with what the frontend currently displays.

        Expected payload shape:
        {
          "physician": "...",
          "billing_model": "FFS"|"PCPCM",
          "entries": [
            {"entry_id":"...", "created_at_utc":"...", "line1_patient":"...", "line2_icd9":"...", "line3_billing":"...", "meta": {...}}
          ]
        }
        """
        dk = _sanitize_date_key(date_key or _local_today_key())
        physician = _clean_str(payload.get("physician"))
        bm = _normalize_model(_clean_str(payload.get("billing_model") or "FFS"))

        entries_in = payload.get("entries") or []
        new_entries: List[BillingEntry] = []
        if isinstance(entries_in, list):
            for it in entries_in:
                if not isinstance(it, dict):
                    continue
                new_entries.append(
                    BillingEntry(
                        entry_id=_clean_str(it.get("entry_id") or uuid.uuid4().hex),
                        created_at_utc=_clean_str(it.get("created_at_utc") or _utcnow_iso()),
                        line1_patient=_clean_str(it.get("line1_patient")),
                        line2_icd9=_clean_str(it.get("line2_icd9")),
                        line3_billing=_clean_str(it.get("line3_billing")),
                        meta=it.get("meta") if isinstance(it.get("meta"), dict) else {},
                    )
                )

        with _LOCK:
            state = self._load(dk)
            state.date_key = dk
            state.physician = physician
            state.billing_model = bm
            state.entries = new_entries
            self._persist(state)
            return state.to_dict()

    def get_print_payload_and_clear(self, date_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a print payload with:
          - ICD-9 descriptions stripped (codes only)
          - Billing descriptions stripped (codes only)
        Then clears that day's saved state (removes file if present).
        """
        dk = _sanitize_date_key(date_key or _local_today_key())
        path = _state_path(self.data_dir, dk)

        with _LOCK:
            state = self._load(dk)

            printable_entries: List[Dict[str, Any]] = []
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
                "billing_model": _normalize_model(state.billing_model),
                "total_patient_count": len(state.entries or []),
                "entries": printable_entries,
                "generated_at_utc": _utcnow_iso(),
            }

            # Clear persisted state
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

            return payload

    def clear_today(self, date_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Clears today's entries (but leaves header defaults).
        """
        dk = _sanitize_date_key(date_key or _local_today_key())
        path = _state_path(self.data_dir, dk)

        with _LOCK:
            state = self._default_state(dk)
            # persist empty state (or remove file); prefer remove for "cleared"
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                # fallback: persist empty file
                self._persist(state)

            return state.to_dict()

    # -------------------------
    # Minimal backward-compat helpers (optional)
    # -------------------------

    def get_daily_text(self, date_key: Optional[str] = None) -> str:
        """
        Derives a text blob from entries (3 lines per entry, separated by blank line).
        """
        state = self._load(date_key)
        blocks: List[str] = []
        for e in state.entries:
            blocks.append("\n".join([
                (e.line1_patient or "").strip(),
                (e.line2_icd9 or "").strip(),
                (e.line3_billing or "").strip(),
            ]).strip())
        return ("\n\n".join([b for b in blocks if b]) + ("\n" if blocks else "")).replace("\r\n", "\n").replace("\r", "\n")
