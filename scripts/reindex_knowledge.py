import json
from app.knowledge_ingest import reindex_all

if __name__ == "__main__":
    result = reindex_all()
    print(json.dumps(result, indent=2))
