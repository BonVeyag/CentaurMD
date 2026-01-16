import os
import sqlite3
import tempfile
from pathlib import Path

from app import knowledge_ingest


def build_temp_index(tmpdir: Path):
    db_path = tmpdir / "knowledge.db"
    knowledge_ingest.DB_PATH = str(db_path)
    knowledge_ingest.DB_HASH_PATH = str(tmpdir / "knowledge.db.sha256")
    knowledge_ingest.INDEX_DIR = str(tmpdir)
    conn = sqlite3.connect(str(db_path))
    knowledge_ingest._ensure_db(conn)

    # Minimal ICD-9
    conn.execute("INSERT INTO icd9_fts(code, description, normalized) VALUES (?,?,?)", ("401.9", "Essential hypertension", "essential hypertension"))

    # Minimal SOMB chunks
    conn.execute(
        "INSERT INTO somb_chunks(chunk_id,text,doc_type,effective_date,filename,page,doc_id) VALUES (?,?,?,?,?,?,?)",
        ("proc#1", "03.03A Office visit code\n", "somb_procedure_list", "2025-03-14", "proc.pdf", 1, "proc.pdf"),
    )
    conn.execute("INSERT INTO somb_fts(text,chunk_id) VALUES (?,?)", ("03.03A Office visit code", "proc#1"))

    conn.execute(
        "INSERT INTO somb_chunks(chunk_id,text,doc_type,effective_date,filename,page,doc_id) VALUES (?,?,?,?,?,?,?)",
        ("rule#1", "03.03A must follow governing rules", "somb_governing_rules", "2025-03-14", "rules.pdf", 2, "rules.pdf"),
    )
    conn.execute("INSERT INTO somb_fts(text,chunk_id) VALUES (?,?)", ("03.03A must follow governing rules", "rule#1"))

    conn.execute(
        "INSERT INTO somb_chunks(chunk_id,text,doc_type,effective_date,filename,page,doc_id) VALUES (?,?,?,?,?,?,?)",
        ("price#1", "03.03A fee $37.50", "somb_price_list", "2025-03-14", "price.pdf", 3, "price.pdf"),
    )
    conn.execute("INSERT INTO somb_fts(text,chunk_id) VALUES (?,?)", ("03.03A fee $37.50", "price#1"))

    conn.commit()
    knowledge_ingest._write_db_hash()


def test_search_icd9_and_somb():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        build_temp_index(td_path)
        res_icd = knowledge_ingest.search_icd9("hypertension", top_k=5)
        assert any(r["code"] == "401.9" for r in res_icd)

        res_proc = knowledge_ingest.search_somb("office visit", top_k=5, doc_type="somb_procedure_list")
        assert any("03.03A" in r["text"] for r in res_proc)

        exact = knowledge_ingest.get_chunks_containing_code("03.03A", doc_type="somb_procedure_list", top_k=2)
        assert exact and exact[0]["doc_type"] == "somb_procedure_list"

