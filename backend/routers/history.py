"""
DeepShield — History Router (JSON store)
GET  /history
GET  /history/{id}
DELETE /history/{id}
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from backend.auth.dependencies import get_current_user
from backend.database import detections_collection, _load, _save

router = APIRouter(prefix="/history", tags=["History"])


@router.get("")
async def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    media_type: Optional[str] = Query(None),
    prediction: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    query: dict = {"user_id": current_user["_id"]}
    if media_type:
        query["media_type"] = media_type.lower()
    if prediction:
        query["prediction"] = prediction.upper()

    col = detections_collection()
    total = col.count_documents(query)
    skip = (page - 1) * limit
    docs = col.find(query, skip=skip, limit=limit)

    # Strip heavy blobs from list view
    items = []
    for doc in docs:
        if "result" in doc:
            r = dict(doc["result"])
            r.pop("heatmap_b64", None)
            r.pop("spectrogram_b64", None)
            doc["result"] = r
        doc["id"] = doc.pop("_id")
        items.append(doc)

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, (total + limit - 1) // limit),
        "items": items,
    }


@router.get("/{detection_id}")
async def get_detection(
    detection_id: str,
    current_user: dict = Depends(get_current_user),
):
    data = _load()
    doc = data["detections"].get(detection_id)
    if not doc or doc.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=404, detail="Detection not found.")
    return {"id": detection_id, **doc}


@router.delete("/{detection_id}")
async def delete_detection(
    detection_id: str,
    current_user: dict = Depends(get_current_user),
):
    data = _load()
    doc = data["detections"].get(detection_id)
    if not doc or doc.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=404, detail="Detection not found.")

    del data["detections"][detection_id]
    _save(data)
    return {"message": "Detection deleted successfully."}
