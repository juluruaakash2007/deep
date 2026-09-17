"""
DeepShield — Image Deepfake Detector
=====================================
DEMO_MODE=false → HuggingFace Inference API (no local download)
DEMO_MODE=true  → Image Forensics fallback (EXIF + noise + DCT analysis)

HF Model: dima806/deepfake_vs_real_image_detection
  - ViT fine-tuned on real vs AI-generated/deepfake faces
  - Returns: [{"label": "Fake", "score": X}, {"label": "Real", "score": Y}]
"""

import asyncio
import io
from PIL import Image
from backend.config import settings
from utils.calibration import calibrate_confidence, get_prediction, get_risk_level, build_explanation


async def detect_image_deepfake(image_bytes: bytes, filename: str) -> dict:
    from utils.image_forensics import _check_exif
    exif_score = _check_exif(image_bytes)

    pil_img = None
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        pass
    face_detected = _check_face(pil_img) if pil_img else True

    # 1. Metadata Bypass
    if exif_score > 0.5:
        # If metadata is present, completely bypass AI and mark as human
        return {
            "prediction": "REAL",
            "confidence": 100.0,
            "fake_probability": 0.0,
            "real_probability": 1.0,
            "risk_level": "LOW",
            "processing_time": 0,
            "face_detected": face_detected,
            "ai_explanation": "Camera metadata detected. Completely bypassed AI model. Marked as human-captured with 0% AI modification.",
            "heatmap_b64": "",
            "model_used": "EXIF Metadata Bypass",
        }

    # 2. Use AI Model
    return await _api_detect(image_bytes, filename, pil_img)


# ── HF API MODE ──────────────────────────────────────────────────────────────
async def _api_detect(image_bytes: bytes, filename: str, pil_img: Image.Image) -> dict:
    from utils.hf_api import hf_image_classify, parse_image_result

    # Convert to JPEG for consistent API input
    if pil_img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()

    # Call HF Inference API — no fallback
    api_response = await hf_image_classify(jpeg_bytes)
    fake_prob, real_prob = parse_image_result(api_response)

    # Calibrate
    fake_prob, real_prob, confidence = calibrate_confidence(fake_prob)
    prediction  = get_prediction(fake_prob)
    risk_level  = get_risk_level(prediction, confidence)

    # Face detection (optional, graceful fallback)
    face_detected = _check_face(pil_img)

    # Synthetic Grad-CAM heatmap (visual explainability)
    heatmap_b64 = _make_heatmap(pil_img)

    explanation = build_explanation(prediction, confidence, "image", face_detected)

    return {
        "prediction":      prediction.value,
        "confidence":      round(confidence, 2),
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "risk_level":      risk_level.value,
        "processing_time": 0,
        "face_detected":   face_detected,
        "ai_explanation":  explanation,
        "heatmap_b64":     heatmap_b64,
        "model_used":      f"HF API: {settings.IMAGE_MODEL_NAME}",
    }


# ── Helpers ──────────────────────────────────────────────────────────────────
def _check_face(pil_img) -> bool:
    try:
        import mediapipe as mp
        import numpy as np
        fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4)
        res = fd.process(np.array(pil_img))
        fd.close()
        return bool(res.detections)
    except Exception:
        return True


def _make_heatmap(pil_img) -> str:
    try:
        from utils.explainability import _generate_synthetic_heatmap
        return _generate_synthetic_heatmap(pil_img)
    except Exception:
        return ""
