"""
DeepShield Backend — Application Settings
==========================================
Uses pydantic-settings to load from .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "DeepShield"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security (simple token auth — no JWT)
    SECRET_SALT: str = "deepshield-salt-change-in-prod"

    # Database (JSON file store)
    DB_PATH: str = "deepshield_db.json"

    # AI — HuggingFace Inference API
    HF_TOKEN: str = ""           # required — set in Render env vars
    HF_API_BASE: str = "https://router.huggingface.co/hf-inference/models"
    IMAGE_MODEL_NAME: str = "capcheck/ai-image-detection"   # ViT-Base, CIFAKE-trained
    AUDIO_MODEL_NAME: str = "HamedAGH/deepfake-audio-detection"
    VIDEO_MODEL: str = "frame_api"   # sends frames to image model

    # File Upload
    UPLOAD_MAX_SIZE_MB: int = 100
    ALLOWED_IMAGE_TYPES: str = "image/jpeg,image/png,image/webp,image/bmp"
    ALLOWED_VIDEO_TYPES: str = "video/mp4,video/avi,video/mov,video/mkv"
    ALLOWED_AUDIO_TYPES: str = "audio/wav,audio/mp3,audio/ogg,audio/flac,audio/mpeg"

    # Reports
    REPORTS_DIR: str = "reports"
    FRONTEND_URL: str = "http://localhost:5500"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # silently skip unknown .env keys
    )

    @property
    def allowed_image_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_IMAGE_TYPES.split(",")]

    @property
    def allowed_video_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_VIDEO_TYPES.split(",")]

    @property
    def allowed_audio_types_list(self) -> list[str]:
        return [t.strip() for t in self.ALLOWED_AUDIO_TYPES.split(",")]


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
