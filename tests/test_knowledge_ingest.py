import os
from app.knowledge_ingest import reindex_all, search_icd9
from app.api import resolve_ffs, ResolveFfsPayload


def test_reindex_counts():
    res = reindex_all()
    assert res.get("icd9_rows", 0) > 1000
    assert res.get("somb_chunks", 0) > 100
    assert os.path.exists(res.get("db_path", ""))


def test_icd9_search_hypertension():
    cands = search_icd9("hypertension", limit=5)
    codes = [c.get("code", "") for c in cands]
    assert any(code.startswith("401") for code in codes)


def test_resolve_ffs_structure():
    payload = ResolveFfsPayload(note_context="office visit for cough and fever", top_k=3)
    res = resolve_ffs(payload)
    assert "knowledge_version" in res
    assert "used_fallback" in res
    assert "suggested" in res
    assert "trace_id" in res
