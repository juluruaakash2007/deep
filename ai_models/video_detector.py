"""
DeepShield — Video Deepfake Detector
=====================================
DEMO_MODE=false → Extracts frames → sends each to HF image API
DEMO_MODE=true  → Seeded simulation

Strategy: Extract up to 8 evenly-spaced frames from the video,
send each to the image deepfake detection API, aggregate scores
with temporal weighting (later frames weighted higher).
"""

import asyncio
import random
import io
import numpy as np
from PIL import Image
from backend.config import settings
from utils.calibration import calibrate_confidence, get_prediction, get_risk_level, build_explanation


async def detect_video_deepfake(video_bytes: bytes, filename: str) -> dict:
    return await _api_detect(video_bytes, filename)


# ── HF API MODE ──────────────────────────────────────────────────────────────
async def _api_detect(video_bytes: bytes, filename: str) -> dict:
    from utils.hf_api import hf_image_classify, parse_image_result

    # Extract frames
    frames = _extract_frames(video_bytes, num_frames=8)

    if not frames:
        raise ValueError("Could not extract frames from video. Make sure cv2 is installed and the video is valid.")

    # Analyse each frame concurrently (up to 4 at a time to respect rate limits)
    semaphore = asyncio.Semaphore(4)

    async def analyse_frame(pil_frame: Image.Image, idx: int) -> dict | None:
        async with semaphore:
            try:
                buf = io.BytesIO()
                pil_frame.save(buf, format="JPEG", quality=85)
                api_resp = await hf_image_classify(buf.getvalue())
                fake_p, real_p = parse_image_result(api_resp)
                fake_p, _, conf = calibrate_confidence(fake_p)
                pred = get_prediction(fake_p)
                return {
                    "frame_index": idx,
                    "timestamp": round(idx / max(len(frames) - 1, 1) * _estimate_duration(video_bytes), 2),
                    "confidence": round(conf, 2),
                    "prediction": pred.value,
                    "fake_prob": fake_p,
                }
            except Exception:
                return None

    results = await asyncio.gather(*[analyse_frame(f, i) for i, f in enumerate(frames)])
    frame_scores = [r for r in results if r is not None]

    if not frame_scores:
        raise ValueError("AI model failed to analyze any frames from this video.")

    # Aggregate: temporal-weighted mean
    raw_probs = [s["fake_prob"] for s in frame_scores]
    weights   = np.linspace(0.6, 1.0, len(raw_probs))
    agg_prob  = float(np.average(raw_probs, weights=weights))

    fake_prob, real_prob, confidence = calibrate_confidence(agg_prob)
    prediction = get_prediction(fake_prob)
    risk_level = get_risk_level(prediction, confidence)

    timeline = [{"t": s["timestamp"], "confidence": s["confidence"], "prediction": s["prediction"]}
                for s in frame_scores]
    explanation = build_explanation(prediction, confidence, "video")

    return {
        "prediction":      prediction.value,
        "confidence":      round(confidence, 2),
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "risk_level":      risk_level.value,
        "processing_time": 0,
        "frames_analyzed": len(frame_scores),
        "frame_scores":    frame_scores,
        "timeline_data":   timeline,
        "ai_explanation":  explanation,
        "model_used":      f"HF API (frame analysis): {settings.IMAGE_MODEL_NAME}",
    }




# ── Helpers ──────────────────────────────────────────────────────────────────
def _extract_frames(video_bytes: bytes, num_frames: int = 8) -> list[Image.Image]:
    """Extract evenly-spaced frames using OpenCV (if available) or return []."""
    try:
        import cv2
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            cap   = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return []
            indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
            frames  = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(cv2.resize(rgb, (224, 224)))
                    frames.append(pil)
            cap.release()
            return frames
        finally:
            os.unlink(tmp_path)
    except Exception:
        return []


def _estimate_duration(video_bytes: bytes) -> float:
    """Rough duration estimate in seconds based on file size."""
    return round(len(video_bytes) / 500_000, 1)  # ~500KB/s compressed video
