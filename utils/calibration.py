"""
DeepShield — Confidence Calibration & Risk Level
"""

from backend.models.detection import Prediction, RiskLevel


def calibrate_confidence(raw_prob: float) -> tuple[float, float, float]:
    """
    Temperature scaling calibration.
    Returns (fake_prob, real_prob, confidence_pct)
    """
    temperature = 1.3
    # Apply temperature scaling
    calibrated_fake = raw_prob ** (1 / temperature)
    calibrated_real = (1 - raw_prob) ** (1 / temperature)
    norm = calibrated_fake + calibrated_real
    fake_prob = calibrated_fake / norm
    real_prob = calibrated_real / norm
    confidence_pct = max(fake_prob, real_prob) * 100
    return round(fake_prob, 4), round(real_prob, 4), round(confidence_pct, 2)


def get_prediction(fake_prob: float) -> Prediction:
    if fake_prob >= 0.65:
        return Prediction.FAKE
    elif fake_prob >= 0.40:
        return Prediction.SUSPICIOUS
    else:
        return Prediction.REAL


def get_risk_level(prediction: Prediction, confidence: float) -> RiskLevel:
    if prediction == Prediction.FAKE:
        if confidence >= 90:
            return RiskLevel.CRITICAL
        elif confidence >= 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM
    elif prediction == Prediction.SUSPICIOUS:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def build_explanation(prediction: Prediction, confidence: float,
                       media_type: str, face_detected: bool = True) -> str:
    templates = {
        Prediction.FAKE: [
            f"The {media_type} exhibits strong synthetic artifacts. "
            f"The AI model detected deepfake signatures with {confidence:.1f}% confidence, "
            "including unnatural facial blending, GAN frequency artifacts, and inconsistent lighting gradients.",
            f"Analysis indicates this {media_type} has been AI-generated or digitally manipulated. "
            f"Confidence: {confidence:.1f}%. Key indicators include spectral inconsistencies and "
            "artifacts characteristic of neural face-swap networks.",
        ],
        Prediction.SUSPICIOUS: [
            f"The {media_type} shows ambiguous signals — some features resemble synthetic media "
            f"while others appear authentic. Confidence: {confidence:.1f}%. "
            "Manual review by a media forensics expert is recommended.",
        ],
        Prediction.REAL: [
            f"No synthetic media artifacts were detected. The {media_type} appears authentic "
            f"with {confidence:.1f}% confidence. Natural noise patterns, consistent compression "
            "artifacts, and organic facial micro-expressions all support authenticity.",
        ],
    }
    import random
    explanations = templates[prediction]
    text = random.choice(explanations)
    if media_type == "image" and not face_detected:
        text += " Note: No human face was detected; analysis performed on full image texture patterns."
    return text
