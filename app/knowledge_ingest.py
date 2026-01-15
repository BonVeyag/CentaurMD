import os
import re
import json
import sqlite3
from typing import List, Dict, Any, Tuple

from app.pdf_utils import extract_pdf_pages
from app.chunk_utils import chunk_text, group_code_rows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ICD9_PATH = os.path.join(ROOT, "knowledge", "icd9", "icd9_dx_long.txt")
SOMB_DIR = os.path.join(ROOT, "knowledge", "alberta_ffs", "somb", "2025-03-14")
SOMB_VERSION = "2025-03-14"
INDEX_DIR = os.path.join(ROOT, "knowledge", "index")
DB_PATH = os.path.join(INDEX_DIR, "knowledge.db")
DB_HASH_PATH = os.path.join(INDEX_DIR, "knowledge.db.sha256")


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS icd9_fts USING fts5(code, description, normalized);")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS somb_chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT,
            doc_type TEXT,
            effective_date TEXT,
            filename TEXT,
            page INT
        );
        """
    )
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS somb_fts USING fts5(text, chunk_id UNINDEXED);")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def ingest_icd9(conn: sqlite3.Connection) -> int:
    if not os.path.exists(ICD9_PATH):
        return 0
    conn.execute("DELETE FROM icd9_fts;")
    rows = 0
    with open(ICD9_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expect format: CODE<tab>DESCRIPTION
            if "\t" in line:
                code, desc = line.split("\t", 1)
            elif " " in line:
                parts = line.split(" ", 1)
                code, desc = parts[0], parts[1]
            else:
                continue
            norm = _normalize(desc)
            conn.execute("INSERT INTO icd9_fts(code, description, normalized) VALUES (?,?,?)", (code.strip(), desc.strip(), norm))
            rows += 1
    conn.commit()
    return rows


def _iter_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    pages = extract_pdf_pages(pdf_path)
    return pages


DOC_TYPE_MAP = {
    "governing_rules": "somb_governing_rules",
    "modifiers": "somb_modifiers",
    "explanatory": "somb_explanatory",
    "price_list": "somb_price_list",
    "procedure_list": "somb_procedure_list",
    "unknown": "unknown",
}


def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    base = "unknown"
    if "governing" in name or "rule" in name:
        base = "governing_rules"
    elif "modifier" in name:
        base = "modifiers"
    elif "explan" in name:
        base = "explanatory"
    elif "price" in name:
        base = "price_list"
    elif "procedure" in name:
        base = "procedure_list"
    return DOC_TYPE_MAP.get(base, "unknown")


def _load_manifest() -> Dict[str, str]:
    manifest_path = os.path.join(SOMB_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect {"files": {"filename.pdf": "doc_type"}}
        if isinstance(data, dict):
            files = data.get("files", {}) or {}
            if isinstance(files, dict):
                out: Dict[str, str] = {}
                for k, v in files.items():
                    base = str(v or "").strip()
                    if base in DOC_TYPE_MAP.values():
                        out[k] = base
                    else:
                        out[k] = DOC_TYPE_MAP.get(base, "unknown")
                return out
    except Exception:
        return {}
    return {}


def _chunk_somb_text(text: str) -> List[str]:
    if not text.strip():
        return []
    # Try to preserve code rows
    rows = group_code_rows(text)
    if rows:
        blocks = rows
    else:
        blocks = [text]
    chunks: List[str] = []
    for block in blocks:
        subchunks = chunk_text(block, min_size=500, max_size=1200)
        for c in subchunks:
            if c.strip():
                chunks.append(c.strip())
    return chunks


def ingest_somb(conn: sqlite3.Connection) -> int:
    somb_raw_dir = os.path.join(SOMB_DIR, "raw", "somb_pdfs")
    if not os.path.isdir(somb_raw_dir):
        return 0
    conn.execute("DELETE FROM somb_chunks;")
    conn.execute("DELETE FROM somb_fts;")
    rows = 0
    manifest_map = _load_manifest()
    for fname in os.listdir(somb_raw_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(somb_raw_dir, fname)
        doc_type = manifest_map.get(fname) or _infer_doc_type(fname)
        pages = _iter_pdf_text(path)
        for page_num, text in pages:
            text = text.strip()
            if not text:
                continue
            for idx, chunk in enumerate(_chunk_somb_text(text)):
                chunk_id = f"{fname}#p{page_num}#c{idx}"
                conn.execute(
                    "INSERT OR REPLACE INTO somb_chunks(chunk_id, text, doc_type, effective_date, filename, page) VALUES (?,?,?,?,?,?)",
                    (chunk_id, chunk, doc_type, "2025-03-14", fname, page_num),
                )
                conn.execute("INSERT INTO somb_fts(text, chunk_id) VALUES (?,?)", (chunk, chunk_id))
                rows += 1
    conn.commit()
    return rows


def _write_db_hash() -> str:
    """
    Compute SHA256 of knowledge DB for provenance.
    """
    try:
        with open(DB_PATH, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        with open(DB_HASH_PATH, "w", encoding="utf-8") as f:
            f.write(h)
        return h
    except Exception:
        return ""


def reindex_all() -> Dict[str, Any]:
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_db(conn)
    icd_rows = ingest_icd9(conn)
    somb_rows = ingest_somb(conn)
    db_hash = _write_db_hash()
    return {"icd9_rows": icd_rows, "somb_chunks": somb_rows, "db_path": DB_PATH, "db_hash": db_hash}


def search_icd9(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    q = _normalize(query)
    cur = conn.execute("SELECT code, description FROM icd9_fts WHERE icd9_fts MATCH ? LIMIT ?", (q, top_k))
    rows = cur.fetchall()
    return [{"code": c, "description": d} for c, d in rows]


def search_somb(query: str, top_k: int = 8, doc_type: str = "") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    match = query or ""
    cur = conn.execute("SELECT text, chunk_id FROM somb_fts WHERE somb_fts MATCH ? LIMIT ?", (match, top_k * 3))
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for text, cid in rows:
        meta = conn.execute("SELECT doc_type, effective_date, filename, page FROM somb_chunks WHERE chunk_id=?", (cid,)).fetchone()
        doc_type_val, eff, fname, page = meta if meta else ("", "", "", 0)
        if doc_type and doc_type_val != doc_type:
            continue
        out.append({"chunk_id": cid, "text": text, "doc_type": doc_type_val, "effective_date": eff, "filename": fname, "page": page})
        if len(out) >= top_k:
            break
    return out


def get_chunks_containing_code(code: str, doc_type: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Return chunks that contain the exact HSC/modifier code string.
    """
    code = (code or "").strip()
    if not code:
        return []
    conn = sqlite3.connect(DB_PATH)
    params: List[Any] = []
    sql = "SELECT chunk_id, text, doc_type, effective_date, filename, page FROM somb_chunks WHERE text LIKE ?"
    params.append(f"%{code}%")
    if doc_type:
        sql += " AND doc_type = ?"
        params.append(doc_type)
    sql += " LIMIT ?"
    params.append(top_k)
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for cid, text, dt, eff, fname, page in rows:
        if not re.search(rf"\\b{re.escape(code)}\\b", text):
            continue
        out.append({"chunk_id": cid, "text": text, "doc_type": dt, "effective_date": eff, "filename": fname, "page": page})
    return out
