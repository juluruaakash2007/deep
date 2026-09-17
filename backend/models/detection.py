"""
DeepShield — Detection Result Pydantic Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class Prediction(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    SUSPICIOUS = "SUSPICIOUS"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FrameScore(BaseModel):
    frame_index: int
    timestamp: float
    confidence: float
    prediction: Prediction


class ImageDetectionResult(BaseModel):
    prediction: Prediction
    confidence: float = Field(..., ge=0, le=100)
    fake_probability: float = Field(..., ge=0, le=1)
    real_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    processing_time: float
    face_detected: bool
    ai_explanation: str
    heatmap_b64: Optional[str] = None   # base64 Grad-CAM heatmap
    model_used: str


class VideoDetectionResult(BaseModel):
    prediction: Prediction
    confidence: float = Field(..., ge=0, le=100)
    fake_probability: float = Field(..., ge=0, le=1)
    real_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    processing_time: float
    frames_analyzed: int
    frame_scores: List[FrameScore] = []
    timeline_data: List[dict] = []
    ai_explanation: str
    model_used: str


class AudioDetectionResult(BaseModel):
    prediction: Prediction
    confidence: float = Field(..., ge=0, le=100)
    fake_probability: float = Field(..., ge=0, le=1)
    real_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    processing_time: float
    duration_seconds: float
    spectrogram_b64: Optional[str] = None
    waveform_data: List[float] = []
    ai_explanation: str
    model_used: str


class DetectionHistory(BaseModel):
    id: str
    user_id: str
    media_type: MediaType
    filename: str
    file_size: int
    prediction: Prediction
    confidence: float
    risk_level: RiskLevel
    processing_time: float
    result: Any  # Full result dict
    created_at: datetime
