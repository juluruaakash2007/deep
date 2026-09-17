"""
DeepShield — Analytics Router (JSON store)
GET /analytics  →  returns dashboard-ready stats
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends
from backend.auth.dependencies import get_current_user
from backend.database import detections_collection

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("")
async def get_analytics(current_user: dict = Depends(get_current_user)):
    col = detections_collection()
    user_id = current_user["_id"]
    stats = col.aggregate_stats(user_id)

    total = stats["total"]
    fake_count       = stats.get("fake", 0)
    real_count       = stats.get("real", 0)
    suspicious_count = stats.get("suspicious", 0)
    avg_conf = round(stats["sum_confidence"] / total, 2) if total > 0 else 0.0

    # ── Daily scans for last 7 days ──────────────────────────────────────────
    today = datetime.utcnow().date()
    day_buckets = {}
    for i in range(6, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        day_buckets[d] = 0

    for doc in stats["recent"]:
        ts = doc.get("created_at", "")
        day = str(ts)[:10]  # "YYYY-MM-DD"
        if day in day_buckets:
            day_buckets[day] += 1

    daily_scans = [{"date": d, "count": c} for d, c in day_buckets.items()]

    # ── Avg confidence by type ───────────────────────────────────────────────
    type_sums   = {"image": 0.0, "video": 0.0, "audio": 0.0}
    type_counts = {"image": 0,   "video": 0,   "audio": 0}
    for doc in stats["recent"]:
        mt = doc.get("media_type", "").lower()
        c  = doc.get("confidence", 0)
        if mt in type_sums:
            type_sums[mt]   += c
            type_counts[mt] += 1

    avg_conf_by_type = {
        t: round(type_sums[t] / type_counts[t], 1) if type_counts[t] else 0.0
        for t in type_sums
    }

    # ── High-risk recent items ───────────────────────────────────────────────
    high_risk = [
        {
            "id":         doc.get("_id", ""),
            "filename":   doc.get("filename", ""),
            "risk_level": doc.get("risk_level", ""),
            "prediction": doc.get("prediction", ""),
            "confidence": doc.get("confidence", 0),
            "created_at": doc.get("created_at", ""),
        }
        for doc in stats["recent"]
        if doc.get("risk_level", "").upper() in ("HIGH", "CRITICAL")
    ][-6:]

    return {
        # KPIs
        "total_scans":      total,
        "fake_count":       fake_count,
        "real_count":       real_count,
        "suspicious_count": suspicious_count,
        "avg_confidence":   avg_conf,
        "detection_rate":   round(fake_count / total * 100, 1) if total > 0 else 0.0,

        # Charts
        "by_media_type": {
            "image": stats.get("image", 0),
            "video": stats.get("video", 0),
            "audio": stats.get("audio", 0),
        },
        "by_risk_level": {
            "high":   stats.get("high", 0),
            "medium": stats.get("medium", 0),
            "low":    stats.get("low", 0),
        },
        "daily_scans":           daily_scans,
        "avg_confidence_by_type": avg_conf_by_type,
        "high_risk_recent":      high_risk,
    }
