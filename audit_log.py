import json
import os
from datetime import datetime, timezone

LOG_FILE = os.path.join(os.path.dirname(__file__), "audit_log.jsonl")


def write_log_entry(entry: dict) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **entry,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_log(limit: int = 50) -> list:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    entries = [json.loads(l) for l in lines]
    return list(reversed(entries[-limit:]))
