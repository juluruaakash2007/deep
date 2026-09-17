"""
DeepShield — JSON File Database
================================
Zero-dependency storage engine.
Data is saved in `deepshield_db.json` in the project root.

Schema:
{
  "users":      { "<user_id>": { ...user doc } },
  "detections": { "<detection_id>": { ...detection doc } }
}
"""

import json
import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Any
import threading

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "deepshield_db.json")
_lock = threading.Lock()

# ── Internal helpers ─────────────────────────────────────────────────────────

def _load() -> dict:
    if not os.path.exists(DB_PATH):
        return {"users": {}, "detections": {}}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _new_id() -> str:
    return uuid.uuid4().hex


# ── Startup / Shutdown (no-ops for JSON store) ───────────────────────────────

async def connect_to_mongo() -> None:
    """No-op — JSON store needs no connection."""
    # Ensure file exists
    with _lock:
        if not os.path.exists(DB_PATH):
            _save({"users": {}, "detections": {}})
    import logging
    logging.getLogger(__name__).info("JSON file store ready: %s", DB_PATH)


async def close_mongo_connection() -> None:
    pass  # Nothing to close


# ── Users ────────────────────────────────────────────────────────────────────

class UsersCollection:
    def find_one(self, query: dict) -> Optional[dict]:
        with _lock:
            data = _load()
        for uid, user in data["users"].items():
            if _matches(user, query):
                return {**user, "_id": uid}
        return None

    def insert_one(self, doc: dict) -> str:
        uid = _new_id()
        with _lock:
            data = _load()
            data["users"][uid] = _serialize(doc)
            _save(data)
        return uid

    def update_one(self, query: dict, update: dict) -> None:
        with _lock:
            data = _load()
            for uid, user in data["users"].items():
                if _matches(user, query) or query.get("_id") == uid:
                    if "$inc" in update:
                        for k, v in update["$inc"].items():
                            user[k] = user.get(k, 0) + v
                    if "$set" in update:
                        user.update(update["$set"])
                    data["users"][uid] = user
                    break
            _save(data)


# ── Detections ───────────────────────────────────────────────────────────────

class DetectionsCollection:
    def count_documents(self, query: dict) -> int:
        with _lock:
            data = _load()
        return sum(1 for d in data["detections"].values() if _matches(d, query))

    def find_one(self, query: dict) -> Optional[dict]:
        with _lock:
            data = _load()
        for did, doc in data["detections"].items():
            if _matches(doc, query):
                return {**doc, "_id": did}
        return None

    def insert_one(self, doc: dict) -> str:
        did = _new_id()
        with _lock:
            data = _load()
            data["detections"][did] = _serialize(doc)
            _save(data)
        return did

    def delete_one(self, query: dict) -> bool:
        with _lock:
            data = _load()
            for did, doc in list(data["detections"].items()):
                if _matches(doc, query):
                    del data["detections"][did]
                    _save(data)
                    return True
        return False

    def find(self, query: dict, sort_by: str = "created_at",
             skip: int = 0, limit: int = 10, reverse: bool = True) -> list:
        with _lock:
            data = _load()
        docs = [
            {**doc, "_id": did}
            for did, doc in data["detections"].items()
            if _matches(doc, query)
        ]
        docs.sort(key=lambda d: d.get(sort_by, ""), reverse=reverse)
        return docs[skip: skip + limit]

    def aggregate_stats(self, user_id: str) -> dict:
        """Return aggregated analytics stats for a user."""
        with _lock:
            data = _load()
        docs = [d for d in data["detections"].values() if d.get("user_id") == user_id]

        total = len(docs)
        stats = {
            "total": total,
            "fake": 0, "real": 0, "suspicious": 0,
            "image": 0, "video": 0, "audio": 0,
            "high": 0, "medium": 0, "low": 0,
            "sum_confidence": 0.0,
            "recent": [],
        }
        for d in docs:
            p = d.get("prediction", "").lower()
            m = d.get("media_type", "").lower()
            r = d.get("risk_level", "").upper()
            stats[p] = stats.get(p, 0) + 1
            stats[m] = stats.get(m, 0) + 1
            if r in ("HIGH", "CRITICAL"):
                stats["high"] += 1
            elif r == "MEDIUM":
                stats["medium"] += 1
            else:
                stats["low"] += 1
            stats["sum_confidence"] += d.get("confidence", 0)

        # Recent 30 (sorted)
        sorted_docs = sorted(docs, key=lambda d: d.get("created_at", ""))
        stats["recent"] = sorted_docs[-30:]
        return stats


# ── Helpers ──────────────────────────────────────────────────────────────────

def _matches(doc: dict, query: dict) -> bool:
    """Check if doc satisfies all query conditions."""
    for key, val in query.items():
        if key == "_id":
            continue
        if isinstance(val, dict):
            # Support simple operators: $in, $gt, $lt
            doc_val = doc.get(key)
            if "$in" in val and doc_val not in val["$in"]:
                return False
        else:
            if doc.get(key) != val:
                return False
    return True


def _serialize(doc: dict) -> dict:
    """Convert datetime objects to ISO strings for JSON storage."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = _serialize(v)
        else:
            out[k] = v
    return out


# ── Singletons ───────────────────────────────────────────────────────────────
_users = UsersCollection()
_detections = DetectionsCollection()


def users_collection() -> UsersCollection:
    return _users


def detections_collection() -> DetectionsCollection:
    return _detections


def get_db():
    return None  # Not needed for JSON store
