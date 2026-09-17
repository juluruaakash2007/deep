"""
DeepShield — Audio Preprocessor
Loads audio, resamples to 16kHz, normalizes waveform.
"""

import numpy as np
import io
import tempfile
import os
from typing import Tuple


def load_audio(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes, resample to target_sr.
    Returns (waveform, sample_rate) where waveform is 1D float32 array.
    """
    try:
        import librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            waveform, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
            return waveform.astype(np.float32), sr
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        raise RuntimeError(f"Audio loading failed: {e}")


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Peak normalize to [-1, 1]."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        return waveform / peak
    return waveform


def get_waveform_data(waveform: np.ndarray, max_points: int = 200) -> list:
    """Downsample waveform to max_points for frontend chart."""
    if len(waveform) <= max_points:
        return waveform.tolist()
    indices = np.linspace(0, len(waveform) - 1, max_points, dtype=int)
    return waveform[indices].tolist()


def get_duration(waveform: np.ndarray, sr: int) -> float:
    return round(len(waveform) / sr, 2)
