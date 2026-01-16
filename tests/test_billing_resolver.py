import json
from app.api import resolve_ffs, ResolveFfsPayload
from app import knowledge_ingest
import tempfile
from pathlib import Path
import sqlite3


def _build_min_index(tmpdir: Path):
    db_path = tmpdir / "knowledge.db"
    knowledge_ingest.DB_PATH = str(db_path)
    knowledge_ingest.DB_HASH_PATH = str(tmpdir / "knowledge.db.sha256")
    knowledge_ingest.INDEX_DIR = str(tmpdir)
    conn = sqlite3.connect(str(db_path))
    knowledge_ingest._ensure_db(conn)
    conn.execute("INSERT INTO icd9_fts(code, description, normalized) VALUES (?,?,?)", ("462", "Acute pharyngitis", "acute pharyngitis"))
    conn.execute(
        "INSERT INTO somb_chunks(chunk_id,text,doc_type,effective_date,filename,page,doc_id) VALUES (?,?,?,?,?,?,?)",
        ("proc#1", "03.03A Office visit", "somb_procedure_list", "2025-03-14", "proc.pdf", 1, "proc.pdf"),
    )
    conn.execute("INSERT INTO somb_fts(text,chunk_id) VALUES (?,?)", ("03.03A Office visit", "proc#1"))
    conn.execute(
        "INSERT INTO somb_chunks(chunk_id,text,doc_type,effective_date,filename,page,doc_id) VALUES (?,?,?,?,?,?,?)",
        ("rule#1", "03.03A must comply governing rule", "somb_governing_rules", "2025-03-14", "rules.pdf", 2, "rules.pdf"),
    )
    conn.execute("INSERT INTO somb_fts(text,chunk_id) VALUES (?,?)", ("03.03A must comply governing rule", "rule#1"))
    conn.execute(
        "INSERT INTO somb_chunks(chunk_id,text,doc_type,effective_date,filename,page,doc_id) VALUES (?,?,?,?,?,?,?)",
        ("price#1", "03.03A fee $37.50", "somb_price_list", "2025-03-14", "price.pdf", 3, "price.pdf"),
    )
    conn.execute("INSERT INTO somb_fts(text,chunk_id) VALUES (?,?)", ("03.03A fee $37.50", "price#1"))
    conn.commit()
    knowledge_ingest._write_db_hash()


def test_resolve_ffs_happy_path(monkeypatch, tmp_path):
    _build_min_index(tmp_path)

    class DummyFacts:
        def __init__(self):
            self.calls = 0

        def __call__(self, note_text: str, background: str):
            self.calls += 1
            return {
                "facts": {
                    "diagnoses": ["sore throat"],
                    "procedures": ["office visit"],
                    "visit_type": "in_person",
                    "duration_minutes": None,
                    "red_flags": [],
                },
                "meta": {"model": "test-model", "prompt_version": "vtest"},
                "review_required": False,
            }

    dummy = DummyFacts()
    monkeypatch.setattr("app.api.extract_billing_facts", dummy)
    monkeypatch.setattr("app.api.KNOWLEDGE_VERSION", "TEST_KV")

    payload = ResolveFfsPayload(note_context="Office visit for sore throat lasting 2 days", top_k=5)
    resp = resolve_ffs(payload)
    assert resp["review_required"] is False
    assert resp["suggested"]["billing_lines"]
    code = resp["suggested"]["billing_lines"][0]
    assert resp["evidence"]["codes"].get(code)
    assert resp["knowledge_version"] == "TEST_KV"


def test_resolve_ffs_missing_rules(monkeypatch, tmp_path):
    _build_min_index(tmp_path)
    # wipe rules
    conn = sqlite3.connect(knowledge_ingest.DB_PATH)
    conn.execute("DELETE FROM somb_chunks WHERE doc_type='somb_governing_rules'")
    conn.execute("DELETE FROM somb_fts WHERE chunk_id NOT IN (SELECT chunk_id FROM somb_chunks)")
    conn.commit()

    class DummyFacts:
        def __call__(self, note_text: str, background: str):
            return {
                "facts": {
                    "diagnoses": ["sore throat"],
                    "procedures": ["office visit"],
                    "visit_type": "in_person",
                    "duration_minutes": None,
                    "red_flags": [],
                },
                "meta": {"model": "test-model", "prompt_version": "vtest"},
                "review_required": False,
            }

    monkeypatch.setattr("app.api.extract_billing_facts", DummyFacts())
    monkeypatch.setattr("app.api.KNOWLEDGE_VERSION", "TEST_KV")

    payload = ResolveFfsPayload(note_context="Office visit for sore throat lasting 2 days", top_k=5)
    resp = resolve_ffs(payload)
    assert resp["review_required"] is True
    assert "missing_governing_rules" in resp["missing_evidence"]
    assert not resp["suggested"]["billing_lines"]

