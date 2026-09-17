"""
DeepShield — Detection Router (JSON store)
POST /detect/image
POST /detect/video
POST /detect/audio
"""

import time
import logging
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from backend.auth.dependencies import get_current_user
from backend.database import detections_collection, users_collection
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Detection"])


def _validate_file(file: UploadFile, allowed_types: list[str]):
    ct = (file.content_type or "").lower()
    # Some browsers send 'application/octet-stream' for all files — allow it
    if ct in ("application/octet-stream", ""):
        return  # rely on client-side type filter
    if ct and ct not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {ct}. Allowed: {allowed_types}",
        )


def _save_detection(user_id: str, media_type: str, filename: str,
                    file_size: int, result: dict) -> str:
    doc = {
        "user_id": user_id,
        "media_type": media_type,
        "filename": filename,
        "file_size": file_size,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "processing_time": result["processing_time"],
        "result": result,
        "created_at": datetime.utcnow().isoformat(),
    }
    col = detections_collection()
    detection_id = col.insert_one(doc)

    # Increment total_scans
    users_collection().update_one({"_id": user_id}, {"$inc": {"total_scans": 1}})
    return detection_id


# ─── Image ───────────────────────────────────────────────────────────────────
@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    _validate_file(file, settings.allowed_image_types_list)
    contents = await file.read()

    if len(contents) > settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")

    from ai_models.image_detector import detect_image_deepfake
    start = time.perf_counter()
    try:
        result = await detect_image_deepfake(contents, file.filename or "image.jpg")
    except Exception as e:
        logger.error(f"Image detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    result["processing_time"] = round(time.perf_counter() - start, 3)
    detection_id = _save_detection(
        user_id=current_user["_id"],
        media_type="image",
        filename=file.filename or "image.jpg",
        file_size=len(contents),
        result=result,
    )
    result["detection_id"] = detection_id
    return result


# ─── Video ───────────────────────────────────────────────────────────────────
@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    _validate_file(file, settings.allowed_video_types_list)
    contents = await file.read()

    if len(contents) > settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")

    from ai_models.video_detector import detect_video_deepfake
    start = time.perf_counter()
    try:
        result = await detect_video_deepfake(contents, file.filename or "video.mp4")
    except Exception as e:
        logger.error(f"Video detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    result["processing_time"] = round(time.perf_counter() - start, 3)
    detection_id = _save_detection(
        user_id=current_user["_id"],
        media_type="video",
        filename=file.filename or "video.mp4",
        file_size=len(contents),
        result=result,
    )
    result["detection_id"] = detection_id
    return result


# ─── Audio ───────────────────────────────────────────────────────────────────
@router.post("/audio")
async def detect_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    _validate_file(file, settings.allowed_audio_types_list)
    contents = await file.read()

    if len(contents) > settings.UPLOAD_MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")

    from ai_models.audio_detector import detect_audio_deepfake
    start = time.perf_counter()
    try:
        result = await detect_audio_deepfake(contents, file.filename or "audio.wav")
    except Exception as e:
        logger.error(f"Audio detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    result["processing_time"] = round(time.perf_counter() - start, 3)
    detection_id = _save_detection(
        user_id=current_user["_id"],
        media_type="audio",
        filename=file.filename or "audio.wav",
        file_size=len(contents),
        result=result,
    )
    result["detection_id"] = detection_id
    return result
