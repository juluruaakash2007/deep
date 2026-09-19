"""
DeepShield — Image Deepfake Detector
=====================================
Logic:
  1. If image has EXIF metadata → completely bypass AI, mark as REAL (0% AI).
  2. If no EXIF → send to HuggingFace Inference API (capcheck/ai-image-detection).

No fallbacks. If the API call fails, the error propagates as-is.

HF Model: capcheck/ai-image-detection
  - ViT-Base fine-tuned on CIFAKE dataset
  - Returns: [{"label": "Fake", "score": X}, {"label": "Real", "score": Y}]
"""

import io
from PIL import Image
from utils.image_forensics import _check_exif
from utils.calibration import calibrate_confidence, get_prediction, get_risk_level, build_explanation
from backend.config import settings


async def detect_image_deepfake(image_bytes: bytes, filename: str) -> dict:
    # ── Step 1: EXIF metadata check ──────────────────────────────
    exif_score = _check_exif(image_bytes)

    if exif_score > 0.5:
        # Camera EXIF detected — completely bypass AI, mark as human-captured
        pil_img = _open_image(image_bytes)
        face_detected = _check_face(pil_img) if pil_img else True
        return {
            "prediction":       "REAL",
            "confidence":       100.0,
            "fake_probability": 0.0,
            "real_probability": 1.0,
            "risk_level":       "LOW",
            "processing_time":  0,
            "face_detected":    face_detected,
            "ai_explanation":   (
                "Camera EXIF metadata detected (Make, Model, DateTime, GPS). "
                "AI analysis completely bypassed — image is marked as human-captured "
                "with 0% AI modification probability."
            ),
            "heatmap_b64":  "",
            "model_used":   "EXIF Metadata Bypass",
        }

    # ── Step 2: No EXIF — use AI model ───────────────────────────
    pil_img = _open_image(image_bytes)
    return await _api_detect(image_bytes, filename, pil_img)


# ── HF API call ───────────────────────────────────────────────────────────────
async def _api_detect(image_bytes: bytes, filename: str, pil_img: Image.Image | None) -> dict:
    from utils.hf_api import hf_image_classify, parse_image_result

    # Convert to JPEG for consistent API input
    if pil_img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()

    # Call HF Inference API — raises HTTPException on any failure (no fallback)
    api_response = await hf_image_classify(jpeg_bytes)
    fake_prob, real_prob = parse_image_result(api_response)

    fake_prob, real_prob, confidence = calibrate_confidence(fake_prob)
    prediction  = get_prediction(fake_prob)
    risk_level  = get_risk_level(prediction, confidence)
    face_detected = _check_face(pil_img)
    heatmap_b64   = _make_heatmap(pil_img)
    explanation   = build_explanation(prediction, confidence, "image", face_detected)

    return {
        "prediction":       prediction.value,
        "confidence":       round(confidence, 2),
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "risk_level":       risk_level.value,
        "processing_time":  0,
        "face_detected":    face_detected,
        "ai_explanation":   explanation,
        "heatmap_b64":      heatmap_b64,
        "model_used":       f"HF API: {settings.IMAGE_MODEL_NAME}",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _open_image(image_bytes: bytes) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _check_face(pil_img: Image.Image | None) -> bool:
    try:
        import mediapipe as mp
        import numpy as np
        fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4)
        res = fd.process(np.array(pil_img))
        fd.close()
        return bool(res.detections)
    except Exception:
        return True


def _make_heatmap(pil_img: Image.Image | None) -> str:
    try:
        from utils.explainability import _generate_synthetic_heatmap
        return _generate_synthetic_heatmap(pil_img)
    except Exception:
        return ""
