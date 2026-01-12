from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from threading import Lock as ThreadLock
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("centaurweb.note_log")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "note_log.sqlite")

NOTE_LOG_TTL_DAYS = 7
DB_LOCK = ThreadLock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_log (
            id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            expires_at_utc TEXT NOT NULL,
            created_by TEXT NOT NULL,
            patient_id TEXT,
            patient_name TEXT,
            chief_complaint TEXT,
            module_types TEXT,
            source_session_id TEXT,
            entry_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_log_created ON note_log(created_at_utc DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_log_expires ON note_log(expires_at_utc)"
    )
    conn.commit()


def _get_db() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _cleanup_expired(conn: sqlite3.Connection) -> None:
    now_iso = _iso(_utc_now())
    conn.execute("DELETE FROM note_log WHERE expires_at_utc <= ?", (now_iso,))
    conn.commit()


def create_note_log_entry(entry: Dict[str, Any]) -> str:
    now = _utc_now()
    entry_id = str(entry.get("id") or uuid4())
    created_at = entry.get("created_at_utc") or _iso(now)
    expires_at = entry.get("expires_at_utc") or _iso(now + timedelta(days=NOTE_LOG_TTL_DAYS))
    created_by = (entry.get("created_by_user_id") or "").strip()
    patient_id = (entry.get("patient_id") or "").strip()
    patient_name = (entry.get("patient_name") or "").strip()
    chief = (entry.get("chief_complaint") or "").strip()
    source_session_id = (entry.get("source_session_id") or "").strip()
    module_types = entry.get("module_types") or []
    if isinstance(module_types, str):
        module_types = [module_types]
    module_types_json = json.dumps(module_types, ensure_ascii=True)

    entry_payload = {
        **entry,
        "id": entry_id,
        "created_at_utc": created_at,
        "expires_at_utc": expires_at,
    }
    entry_json = json.dumps(entry_payload, ensure_ascii=True)

    with DB_LOCK:
        conn = _get_db()
        _cleanup_expired(conn)
        conn.execute(
            """
            INSERT INTO note_log (
                id, created_at_utc, expires_at_utc, created_by, patient_id,
                patient_name, chief_complaint, module_types, source_session_id, entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                created_at,
                expires_at,
                created_by,
                patient_id,
                patient_name,
                chief,
                module_types_json,
                source_session_id,
                entry_json,
            ),
        )
        conn.commit()
        conn.close()

    return entry_id


def list_note_log_entries(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with DB_LOCK:
        conn = _get_db()
        _cleanup_expired(conn)
        rows = conn.execute(
            """
            SELECT id, created_at_utc, patient_name, chief_complaint, module_types, source_session_id
            FROM note_log
            ORDER BY created_at_utc DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        conn.close()

    out: List[Dict[str, Any]] = []
    for row in rows:
        module_types = []
        raw_types = row["module_types"] or "[]"
        try:
            module_types = json.loads(raw_types)
        except Exception:
            module_types = [t.strip() for t in raw_types.split(",") if t.strip()]
        out.append(
            {
                "id": row["id"],
                "created_at_utc": row["created_at_utc"],
                "patient_name": row["patient_name"] or "",
                "chief_complaint": row["chief_complaint"] or "",
                "module_types": module_types,
                "source_session_id": row["source_session_id"] or "",
            }
        )
    return out


def get_note_log_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    with DB_LOCK:
        conn = _get_db()
        _cleanup_expired(conn)
        row = conn.execute(
            "SELECT entry_json FROM note_log WHERE id = ?",
            (entry_id,),
        ).fetchone()
        conn.close()

    if not row:
        return None
    try:
        return json.loads(row["entry_json"] or "{}")
    except Exception:
        logger.warning("Note log entry JSON parse failed for %s", entry_id)
        return None


def delete_note_log_entry(entry_id: str) -> bool:
    with DB_LOCK:
        conn = _get_db()
        _cleanup_expired(conn)
        cur = conn.execute("DELETE FROM note_log WHERE id = ?", (entry_id,))
        conn.commit()
        conn.close()
    return cur.rowcount > 0
