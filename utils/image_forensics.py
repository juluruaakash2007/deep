"""
DeepShield — Image Forensics Heuristics
=========================================
Used as fallback when HF API is unavailable.
Analyses real image properties to distinguish camera-captured
photos from AI-generated / deepfake images.

Heuristics used:
  1. EXIF metadata (camera phones embed GPS, make, model, datetime)
  2. Sensor noise variance  (real photos: moderate noise)
  3. JPEG DCT block artifacts (real photos: consistent 8x8 patterns)
  4. Color channel correlation (real photos: high RGB correlation)
  5. Edge sharpness distribution (AI images: unnaturally uniform)
"""

import io
import numpy as np
from PIL import Image


def analyse_image_authenticity(image_bytes: bytes) -> dict:
    """
    Returns a dict with:
      real_score  : float in [0, 1]  — 1.0 means very likely real
      fake_score  : float in [0, 1]
      signals     : dict of individual evidence signals
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"real_score": 0.5, "fake_score": 0.5, "signals": {}}

    signals = {}

    # ── 1. EXIF metadata ──────────────────────────────────────
    exif_score = _check_exif(image_bytes)
    signals["exif"] = exif_score  # 0=no exif, 0.6=partial, 1.0=full camera exif

    # ── 2. Sensor noise ───────────────────────────────────────
    noise_score = _check_noise(img)
    signals["noise"] = noise_score  # real camera: ~0.7-0.9

    # ── 3. JPEG DCT consistency ───────────────────────────────
    dct_score = _check_dct_consistency(img)
    signals["dct"] = dct_score  # real photos: high consistency

    # ── 4. Color channel correlation ──────────────────────────
    color_score = _check_color_correlation(img)
    signals["color"] = color_score  # real photos: natural RGB correlation

    # ── 5. Edge uniformity (AI tends to be "too smooth") ──────
    edge_score = _check_edge_distribution(img)
    signals["edges"] = edge_score  # real=varied, AI=uniform

    # ── Weighted combination ──────────────────────────────────
    # EXIF is the strongest signal
    weights = {
        "exif":   0.40,
        "noise":  0.20,
        "dct":    0.15,
        "color":  0.15,
        "edges":  0.10,
    }
    real_score = sum(weights[k] * signals[k] for k in weights)
    real_score = float(np.clip(real_score, 0.02, 0.98))
    fake_score = 1.0 - real_score

    return {"real_score": real_score, "fake_score": fake_score, "signals": signals}


# ── EXIF Check ────────────────────────────────────────────────
def _check_exif(image_bytes: bytes) -> float:
    """
    Camera phones embed rich EXIF: Make, Model, DateTime, GPS, focal length.
    AI-generated images typically have NO EXIF or minimal EXIF.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif() if hasattr(img, '_getexif') else None

        if not exif_data:
            # Try alternate EXIF access
            info = img.info or {}
            if "exif" in info and len(info["exif"]) > 50:
                return 0.55  # Has some EXIF bytes
            return 0.10  # No EXIF → likely AI-generated

        # Tag IDs for strong camera signals
        MAKE        = 271   # Camera manufacturer
        MODEL       = 272   # Camera model
        DATETIME    = 306   # Date/Time
        GPS_INFO    = 34853 # GPS data
        FOCAL_LEN   = 37386 # Focal length
        FLASH       = 37385 # Flash info
        ISO         = 34855 # ISO speed
        EXPOSURE    = 33434 # Exposure time

        score = 0.10  # Base for having any EXIF
        if exif_data.get(MAKE):    score += 0.20
        if exif_data.get(MODEL):   score += 0.20
        if exif_data.get(DATETIME):score += 0.15
        if exif_data.get(GPS_INFO):score += 0.15
        if exif_data.get(FOCAL_LEN):score += 0.10
        if exif_data.get(ISO):      score += 0.05
        if exif_data.get(EXPOSURE): score += 0.05

        return min(score, 1.0)
    except Exception:
        return 0.25  # Can't read → neutral-ish


# ── Noise Analysis ────────────────────────────────────────────
def _check_noise(img: Image.Image) -> float:
    """
    Real camera sensors produce natural random noise.
    AI images tend to be unnaturally smooth or have structured noise.
    """
    try:
        arr = np.array(img.resize((128, 128)).convert("L"), dtype=np.float32)

        # High-frequency noise via Laplacian-ish filter
        from numpy.lib.stride_tricks import as_strided
        # Simple gradient magnitude
        gy = np.abs(np.diff(arr, axis=0))
        gx = np.abs(np.diff(arr, axis=1))
        noise_var = float(np.var(gy)) + float(np.var(gx))

        # Real camera photos: noise_var typically 50-500
        # AI-generated (smooth): < 30
        # AI-generated (artifacts): > 800
        if noise_var < 20:
            return 0.15   # Too smooth → likely AI
        elif 20 <= noise_var < 40:
            return 0.35
        elif 40 <= noise_var < 600:
            return 0.80   # Natural camera noise range
        else:
            return 0.50   # Very high noise (could be low-light real photo)
    except Exception:
        return 0.50


# ── DCT Block Consistency ─────────────────────────────────────
def _check_dct_consistency(img: Image.Image) -> float:
    """
    Real JPEG photos from cameras have consistent 8x8 DCT block structure.
    Heavily edited or AI images often show block boundary artifacts.
    """
    try:
        gray = np.array(img.resize((128, 128)).convert("L"), dtype=np.float32)

        # Check for 8x8 block variance (JPEG compression artifacts)
        h, w = gray.shape
        bh, bw = h // 8, w // 8
        block_means = []
        for i in range(bh):
            for j in range(bw):
                block = gray[i*8:(i+1)*8, j*8:(j+1)*8]
                block_means.append(np.mean(block))

        block_means = np.array(block_means)
        variance = float(np.var(block_means))

        # Natural images: moderate block variance
        if 50 < variance < 2000:
            return 0.75
        elif variance <= 50:
            return 0.40   # Too uniform
        else:
            return 0.55   # Very varied (still could be real)
    except Exception:
        return 0.55


# ── Color Channel Correlation ─────────────────────────────────
def _check_color_correlation(img: Image.Image) -> float:
    """
    Real photos have natural RGB correlations from real-world lighting.
    Some GAN artifacts manifest as unusual channel correlations.
    """
    try:
        arr = np.array(img.resize((64, 64)), dtype=np.float32)
        r, g, b = arr[:,:,0].flatten(), arr[:,:,1].flatten(), arr[:,:,2].flatten()

        rg_corr = float(np.corrcoef(r, g)[0,1])
        rb_corr = float(np.corrcoef(r, b)[0,1])
        gb_corr = float(np.corrcoef(g, b)[0,1])

        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

        # Real photos: moderate-high correlation (0.6-0.98)
        # AI face images: often unusually perfect correlation (>0.99) or low (<0.3)
        if 0.55 <= avg_corr <= 0.97:
            return 0.80
        elif avg_corr > 0.97:
            return 0.45  # Suspiciously perfect
        else:
            return 0.40  # Low correlation
    except Exception:
        return 0.60


# ── Edge Distribution ─────────────────────────────────────────
def _check_edge_distribution(img: Image.Image) -> float:
    """
    Real photos have a varied distribution of edge strengths.
    AI images can be too smooth in some areas and over-sharpened in others.
    """
    try:
        gray = np.array(img.resize((128, 128)).convert("L"), dtype=np.float32)
        gy = np.abs(np.diff(gray, axis=0))
        gx = np.abs(np.diff(gray, axis=1))
        edges = np.sqrt(gy[:127,:]**2 + gx[:,:127]**2).flatten()

        # Coefficient of variation of edge strengths
        mean_e = float(np.mean(edges)) + 1e-6
        std_e  = float(np.std(edges))
        cv = std_e / mean_e

        # Real photos: high variation in edges (cv > 1.0)
        # AI images: edges tend to be more uniform
        if cv > 1.2:
            return 0.80
        elif 0.7 <= cv <= 1.2:
            return 0.65
        else:
            return 0.35
    except Exception:
        return 0.55
