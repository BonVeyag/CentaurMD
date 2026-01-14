import sys
from app.knowledge_ingest import search_icd9, search_somb

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_billing_query.py <query text>")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    print("ICD-9 candidates:")
    for rec in search_icd9(q, top_k=5):
        print(f"  {rec.get('code')} - {rec.get('description')}")
    print("\nSOMB chunks (all):")
    for rec in search_somb(q, top_k=5):
        loc = f"{rec.get('filename')} p{rec.get('page')}"
        print(f"  [{rec.get('doc_type')}] {loc}: {rec.get('text')[:160].replace('\n',' ')}...")
    print("\nSOMB chunks (governing_rules):")
    for rec in search_somb(q, top_k=3, doc_type="governing_rules"):
        loc = f"{rec.get('filename')} p{rec.get('page')}"
        print(f"  [{rec.get('doc_type')}] {loc}: {rec.get('text')[:160].replace('\n',' ')}...")
