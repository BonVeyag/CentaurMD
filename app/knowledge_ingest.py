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
INDEX_DIR = os.path.join(ROOT, "knowledge", "index")
DB_PATH = os.path.join(INDEX_DIR, "knowledge.db")


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
    pages: List[Tuple[int, str]] = []
    if PyPDF2 is None:
        return pages
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


def _infer_doc_type(filename: str) -> str:
    name = filename.lower()
    if "governing" in name:
        return "governing_rules"
    if "modifier" in name:
        return "modifiers"
    if "explan" in name:
        return "explanatory"
    if "price" in name:
        return "price_list"
    if "procedure" in name:
        return "procedure_list"
    return "unknown"


def ingest_somb(conn: sqlite3.Connection) -> int:
    somb_raw_dir = os.path.join(SOMB_DIR, "raw", "somb_pdfs")
    if not os.path.isdir(somb_raw_dir):
        return 0
    conn.execute("DELETE FROM somb_chunks;")
    conn.execute("DELETE FROM somb_fts;")
    rows = 0
    for fname in os.listdir(somb_raw_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(somb_raw_dir, fname)
        doc_type = _infer_doc_type(fname)
        for page_num, text in _iter_pdf_text(path):
            text = text.strip()
            if not text:
                continue
            chunk_id = f"{fname}#p{page_num}"
            conn.execute(
                "INSERT OR REPLACE INTO somb_chunks(chunk_id, text, doc_type, effective_date, filename, page) VALUES (?,?,?,?,?,?)",
                (chunk_id, text, doc_type, "2025-03-14", fname, page_num),
            )
            conn.execute("INSERT INTO somb_fts(text, chunk_id) VALUES (?,?)", (text, chunk_id))
            rows += 1
    conn.commit()
    return rows


def reindex_all() -> Dict[str, Any]:
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_db(conn)
    icd_rows = ingest_icd9(conn)
    somb_rows = ingest_somb(conn)
    return {"icd9_rows": icd_rows, "somb_chunks": somb_rows, "db_path": DB_PATH}


def search_icd9(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    q = _normalize(query)
    cur = conn.execute("SELECT code, description FROM icd9_fts WHERE icd9_fts MATCH ? LIMIT ?", (q, top_k))
    rows = cur.fetchall()
    return [{"code": c, "description": d} for c, d in rows]


def search_somb(query: str, top_k: int = 8, doc_type: str = "") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    match = query
    if doc_type:
        match = f"{query} {doc_type}"
    cur = conn.execute("SELECT text, chunk_id FROM somb_fts WHERE somb_fts MATCH ? LIMIT ?", (match, top_k))
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for text, cid in rows:
        meta = conn.execute("SELECT doc_type, effective_date, filename, page FROM somb_chunks WHERE chunk_id=?", (cid,)).fetchone()
        doc_type_val, eff, fname, page = meta if meta else ("", "", "", 0)
        out.append({"chunk_id": cid, "text": text, "doc_type": doc_type_val, "effective_date": eff, "filename": fname, "page": page})
    return out
