import json
import os
import threading
import time
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, Optional

USAGE_LOG_DIR = os.path.join("data", "usage_logs")

class UsageLogger:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        os.makedirs(USAGE_LOG_DIR, exist_ok=True)
        self.daily_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def log_event(self, event_type: str, status: int = 200, meta: Optional[Dict[str, Any]] = None) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        entry = {
            "ts": time.time(),
            "type": event_type,
            "status": status,
            "meta": meta or {},
        }
        path = os.path.join(USAGE_LOG_DIR, f"usage_{day}.jsonl")
        with self.lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            daily = self.daily_counts[day]
            daily[f"events_{event_type}"] += 1
            if status >= 400:
                daily[f"errors_{event_type}"] += 1

    def summarize_day(self, day: Optional[str] = None) -> Dict[str, int]:
        target = day or datetime.now().strftime("%Y-%m-%d")
        counts: Dict[str, int] = defaultdict(int)
        path = os.path.join(USAGE_LOG_DIR, f"usage_{target}.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except Exception:
                        continue
                    etype = entry.get("type") or "unknown"
                    counts[f"events_{etype}"] += 1
                    if entry.get("status", 0) >= 400:
                        counts[f"errors_{etype}"] += 1
        if target in self.daily_counts:
            for key, value in self.daily_counts[target].items():
                counts[key] += value
        counts["day"] = 1
        return dict(counts)

usage_logger = UsageLogger()
