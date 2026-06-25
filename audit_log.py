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
    all_entries = [json.loads(l) for l in lines]

    # Build appeal index: content_id → most recent appeal record
    appeals = {}
    for entry in all_entries:
        if entry.get("type") == "appeal":
            appeals[entry["content_id"]] = entry

    # Return only classification entries, annotated with appeal status
    classified = []
    for entry in all_entries:
        if entry.get("type") == "appeal":
            continue
        cid = entry.get("content_id")
        if cid in appeals:
            entry["appeal_filed"] = True
            entry["appeal_status"] = appeals[cid].get("status")
            entry["appeal_id"] = appeals[cid].get("appeal_id")
        else:
            entry["appeal_filed"] = False
        classified.append(entry)

    return list(reversed(classified[-limit:]))


def get_entry(content_id: str) -> dict | None:
    """Return the original classification entry for a content_id, or None if not found."""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("content_id") == content_id and entry.get("type") != "appeal":
                return entry
    return None
