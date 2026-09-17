"""
DeepShield — Audio Deepfake Detector
======================================
DEMO_MODE=false → HuggingFace Inference API (no local download)
DEMO_MODE=true  → Seeded simulation

HF Model: HamedAGH/deepfake-audio-detection
  - Audio classifier for real vs deepfake/synthetic voices
  - Returns: [{"label": "fake", "score": X}, {"label": "real", "score": Y}]
"""

import asyncio
import random
import numpy as np
import io
from backend.config import settings
from utils.calibration import calibrate_confidence, get_prediction, get_risk_level, build_explanation


async def detect_audio_deepfake(audio_bytes: bytes, filename: str) -> dict:
    return await _api_detect(audio_bytes, filename)


# ── HF API MODE ──────────────────────────────────────────────────────────────
async def _api_detect(audio_bytes: bytes, filename: str) -> dict:
    from utils.hf_api import hf_audio_classify, parse_audio_result

    # Call HF Inference API — no fallback
    api_response = await hf_audio_classify(audio_bytes)
    fake_prob, real_prob = parse_audio_result(api_response)

    fake_prob, real_prob, confidence = calibrate_confidence(fake_prob)
    prediction = get_prediction(fake_prob)
    risk_level = get_risk_level(prediction, confidence)

    duration       = _estimate_duration(audio_bytes)
    waveform_data  = _synthetic_waveform(audio_bytes)
    spectrogram_b64 = _try_spectrogram(audio_bytes)
    explanation    = build_explanation(prediction, confidence, "audio")

    return {
        "prediction":       prediction.value,
        "confidence":       round(confidence, 2),
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "risk_level":       risk_level.value,
        "processing_time":  0,
        "duration_seconds": duration,
        "spectrogram_b64":  spectrogram_b64,
        "waveform_data":    waveform_data,
        "ai_explanation":   explanation,
        "model_used":       f"HF API: {settings.AUDIO_MODEL_NAME}",
    }




# ── Helpers ──────────────────────────────────────────────────────────────────
def _estimate_duration(audio_bytes: bytes) -> float:
    """Rough estimate: ~32KB/s for 16kHz mono."""
    return round(len(audio_bytes) / 32_000, 2)


def _synthetic_waveform(audio_bytes: bytes, points: int = 200) -> list:
    """Generate a plausible waveform preview from byte statistics."""
    seed = int(sum(audio_bytes[:256]) % 10000)
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, 4 * np.pi, points)
    wave = (
        np.sin(t) * 0.5
        + np.sin(2.3 * t) * 0.25
        + rng.uniform(-0.15, 0.15, points)
    )
    return np.clip(wave, -1, 1).tolist()


def _try_spectrogram(audio_bytes: bytes) -> str:
    """
    Try to generate a real mel-spectrogram.
    Requires librosa — gracefully returns "" if not installed.
    """
    try:
        import librosa
        import tempfile, os
        from utils.explainability import generate_spectrogram_b64

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
            return generate_spectrogram_b64(waveform, sr)
        finally:
            os.unlink(tmp_path)
    except Exception:
        return ""
