from __future__ import annotations

import csv
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

_LOCK = threading.Lock()

ENV_ICD9_PATH = "CENTAUR_ICD9_PATH"
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "icd9_ab.csv")
DEFAULT_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "icd9_ab.json")

ENV_ICD9_KEYWORDS_PATH = "CENTAUR_ICD9_KEYWORDS_PATH"
DEFAULT_KEYWORDS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "icd9_keywords.json")


@dataclass(frozen=True)
class Icd9Record:
    code: str
    label: str
    synonyms: List[str]
    label_norm: str
    synonyms_norm: List[str]


_LOADED = False
_RECORDS: List[Icd9Record] = []
_CODE_MAP: Dict[str, Icd9Record] = {}
_CODE_KEY_MAP: Dict[str, Icd9Record] = {}
_CODE_NODOT_MAP: Dict[str, Icd9Record] = {}
_KEYWORD_MAP: Optional[Dict[str, List[str]]] = None


def _normalize_text(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _code_key(code: str) -> str:
    return re.sub(r"[^0-9.]", "", (code or "").strip())


def _code_key_nodot(code: str) -> str:
    return re.sub(r"[^0-9]", "", (code or "").strip())


def _split_synonyms(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    return [p.strip() for p in s.split(",") if p.strip()]


def _row_get(row: Dict[str, Any], key: str) -> str:
    for k, v in row.items():
        if str(k).strip().lower() == key:
            return str(v or "").strip()
    return ""


def _load_from_csv(path: str) -> List[Icd9Record]:
    out: List[Icd9Record] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = _row_get(row, "code")
            label = _row_get(row, "label")
            synonyms_raw = _row_get(row, "synonyms")
            if not code or not label:
                continue
            synonyms = _split_synonyms(synonyms_raw)
            label_norm = _normalize_text(label)
            synonyms_norm = [_normalize_text(s) for s in synonyms if s.strip()]
            out.append(Icd9Record(code=code, label=label, synonyms=synonyms, label_norm=label_norm, synonyms_norm=synonyms_norm))
    return out


def _load_from_json(path: str) -> List[Icd9Record]:
    out: List[Icd9Record] = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw: Iterable[Dict[str, Any]]
    if isinstance(data, dict):
        raw = data.get("icd9", []) or data.get("codes", []) or []
    else:
        raw = data or []
    if not isinstance(raw, list):
        return []
    for row in raw:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "").strip()
        label = str(row.get("label") or "").strip()
        synonyms = _split_synonyms(row.get("synonyms"))
        if not code or not label:
            continue
        label_norm = _normalize_text(label)
        synonyms_norm = [_normalize_text(s) for s in synonyms if s.strip()]
        out.append(Icd9Record(code=code, label=label, synonyms=synonyms, label_norm=label_norm, synonyms_norm=synonyms_norm))
    return out


def _discover_path(path_override: Optional[str] = None) -> Optional[str]:
    if path_override:
        return path_override
    env_path = (os.getenv(ENV_ICD9_PATH) or "").strip()
    if env_path:
        return env_path
    if os.path.exists(DEFAULT_CSV_PATH):
        return DEFAULT_CSV_PATH
    if os.path.exists(DEFAULT_JSON_PATH):
        return DEFAULT_JSON_PATH
    return None


def _build_indexes(records: List[Icd9Record]) -> None:
    _RECORDS.clear()
    _CODE_MAP.clear()
    _CODE_KEY_MAP.clear()
    _CODE_NODOT_MAP.clear()

    for rec in records:
        if rec.code in _CODE_MAP:
            continue
        _RECORDS.append(rec)
        _CODE_MAP[rec.code] = rec

        key = _code_key(rec.code)
        if key and key not in _CODE_KEY_MAP:
            _CODE_KEY_MAP[key] = rec

        nodot = _code_key_nodot(rec.code)
        if nodot and nodot not in _CODE_NODOT_MAP:
            _CODE_NODOT_MAP[nodot] = rec


def load_icd9_dictionary(path_override: Optional[str] = None, force: bool = False) -> List[Icd9Record]:
    global _LOADED
    with _LOCK:
        if _LOADED and not force:
            return list(_RECORDS)
        path = _discover_path(path_override)
        if not path:
            _RECORDS.clear()
            _CODE_MAP.clear()
            _CODE_KEY_MAP.clear()
            _CODE_NODOT_MAP.clear()
            _LOADED = True
            return []

        try:
            if path.lower().endswith(".json"):
                records = _load_from_json(path)
            else:
                records = _load_from_csv(path)
        except Exception:
            records = []

        _build_indexes(records)
        _LOADED = True
        return list(_RECORDS)


def reset_icd9_cache() -> None:
    global _LOADED, _KEYWORD_MAP
    with _LOCK:
        _LOADED = False
        _KEYWORD_MAP = None
        _RECORDS.clear()
        _CODE_MAP.clear()
        _CODE_KEY_MAP.clear()
        _CODE_NODOT_MAP.clear()


def get_icd9_by_code(code: str) -> Optional[Dict[str, Any]]:
    load_icd9_dictionary()
    raw = (code or "").strip()
    if not raw:
        return None
    if raw in _CODE_MAP:
        rec = _CODE_MAP[raw]
        return {"code": rec.code, "label": rec.label, "synonyms": rec.synonyms}
    key = _code_key(raw)
    if key and key in _CODE_KEY_MAP:
        rec = _CODE_KEY_MAP[key]
        return {"code": rec.code, "label": rec.label, "synonyms": rec.synonyms}
    nodot = _code_key_nodot(raw)
    if nodot and nodot in _CODE_NODOT_MAP:
        rec = _CODE_NODOT_MAP[nodot]
        return {"code": rec.code, "label": rec.label, "synonyms": rec.synonyms}
    return None


def search_icd9(query: str, limit: int = 20) -> List[Dict[str, str]]:
    load_icd9_dictionary()
    q = _normalize_text(query or "")
    if not q:
        return []
    q_code = _code_key(query)

    scored: List[tuple[int, str, Icd9Record]] = []
    for rec in _RECORDS:
        score = 0
        if q_code:
            if rec.code == q_code:
                score = 1000
            elif rec.code.startswith(q_code):
                score = max(score, 900)
            elif q_code in rec.code:
                score = max(score, 700)

        if rec.label_norm.startswith(q):
            score = max(score, 600)
        elif f" {q}" in f" {rec.label_norm}":
            score = max(score, 400)

        for syn in rec.synonyms_norm:
            if syn.startswith(q):
                score = max(score, 500)
                break
            if f" {q}" in f" {syn}":
                score = max(score, 300)
                break

        if score > 0:
            scored.append((score, rec.code, rec))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[Dict[str, str]] = []
    max_items = max(0, min(int(limit), 50))
    if max_items == 0:
        return []
    for _, _, rec in scored[: max_items]:
        out.append({"code": rec.code, "label": rec.label})
    return out


def _load_keyword_map() -> Dict[str, List[str]]:
    global _KEYWORD_MAP
    with _LOCK:
        if _KEYWORD_MAP is not None:
            return _KEYWORD_MAP

        path = (os.getenv(ENV_ICD9_KEYWORDS_PATH) or "").strip() or DEFAULT_KEYWORDS_PATH
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    _KEYWORD_MAP = {str(k).strip().lower(): [str(c).strip() for c in (v or [])] for k, v in data.items()}
                elif isinstance(data, list):
                    out: Dict[str, List[str]] = {}
                    for row in data:
                        if not isinstance(row, dict):
                            continue
                        kw = str(row.get("keyword") or row.get("term") or "").strip().lower()
                        codes = row.get("codes") or row.get("code") or []
                        codes_list = _split_synonyms(codes)
                        if kw and codes_list:
                            out[kw] = codes_list
                    _KEYWORD_MAP = out
                else:
                    _KEYWORD_MAP = {}
            except Exception:
                _KEYWORD_MAP = {}
        else:
            _KEYWORD_MAP = {
                "hypertension": ["401"],
                "diabetes": ["250"],
                "asthma": ["493"],
                "copd": ["496"],
                "heart failure": ["428.0"],
                "heart failure chronic": ["428.0"],
                "hypothyroid": ["244.9"],
                "hyperthyroid": ["242.9"],
                "hyperlipidemia": ["272.4"],
                "depression": ["311"],
                "anxiety": ["300.00"],
                "obesity": ["278.00"],
                "otitis media": ["382.9"],
                "uti": ["599.0"],
                "urinary tract infection": ["599.0"],
            }
        return _KEYWORD_MAP


def suggest_icd9_from_text(text: str, limit: int = 3) -> List[Dict[str, Any]]:
    load_icd9_dictionary()
    text_norm = _normalize_text(text or "")
    if not text_norm:
        return []

    kw_map = _load_keyword_map()
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for kw, codes in kw_map.items():
        if not kw or kw in seen:
            continue
        pattern = rf"(^|\s){re.escape(kw)}(\s|$)"
        if not re.search(pattern, text_norm):
            continue
        for code in codes:
            rec = get_icd9_by_code(code)
            if not rec:
                continue
            if rec["code"] in seen:
                continue
            out.append({"code": rec["code"], "label": rec["label"]})
            seen.add(rec["code"])
            if len(out) >= limit:
                return out
    return out
