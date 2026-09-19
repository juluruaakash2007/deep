"""
DeepShield — HuggingFace Inference API Client
==============================================
Calls HF-hosted models via REST API.
No local model downloads — inference runs on HuggingFace servers.

Image model : capcheck/ai-image-detection  (ViT-Base, CIFAKE-trained)
Audio model : HamedAGH/deepfake-audio-detection

No fallbacks. All errors propagate as HTTP exceptions.
"""

import httpx
import asyncio
import logging
from fastapi import HTTPException
from backend.config import settings

logger = logging.getLogger(__name__)

HF_TIMEOUT   = 60.0  # seconds — HF cold-start can take ~20s
HF_MAX_RETRY = 2      # retry on 503 (model still loading)


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.HF_TOKEN}"}


async def hf_image_classify(image_bytes: bytes, model: str | None = None) -> list[dict]:
    """
    POST raw image bytes to HF image-classification endpoint.
    Model: capcheck/ai-image-detection
    Returns: [{"label": "Fake", "score": 0.95}, {"label": "Real", "score": 0.05}]
    Raises HTTPException on missing token, DNS failure, or timeout.
    """
    if not settings.HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not configured. Set it as an environment variable on your hosting platform."
        )

    model = model or settings.IMAGE_MODEL_NAME
    url   = f"{settings.HF_API_BASE}/{model}"

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            for attempt in range(HF_MAX_RETRY + 1):
                resp = await client.post(url, headers=_headers(), content=image_bytes)

                if resp.status_code == 200:
                    return resp.json()

                # 503 = model still loading on HF side — wait and retry
                if resp.status_code == 503 and attempt < HF_MAX_RETRY:
                    wait = resp.json().get("estimated_time", 20)
                    logger.info(f"HF image model loading, waiting {wait}s…")
                    await asyncio.sleep(min(float(wait), 30))
                    continue

                logger.error(f"HF API error {resp.status_code}: {resp.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"HuggingFace API returned {resp.status_code}: {resp.text[:200]}"
                )

    except httpx.ConnectError as e:
        logger.error(f"HF API DNS/network error: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot reach HuggingFace API — DNS resolution failed. "
                "Ensure HF_TOKEN is set on your hosting platform and the server "
                f"has outbound internet access. Error: {e}"
            )
        )
    except httpx.TimeoutException as e:
        logger.error(f"HF API timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API timed out. The model may be cold-starting; please retry in 30 seconds."
        )

    raise HTTPException(status_code=502, detail="HF API: max retries exceeded.")


async def hf_audio_classify(audio_bytes: bytes, model: str | None = None) -> list[dict]:
    """
    POST raw audio bytes to HF audio-classification endpoint.
    Model: HamedAGH/deepfake-audio-detection
    Returns: [{"label": "fake", "score": 0.88}, {"label": "real", "score": 0.12}]
    Raises HTTPException on missing token, DNS failure, or timeout.
    """
    if not settings.HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not configured. Set it as an environment variable on your hosting platform."
        )

    model = model or settings.AUDIO_MODEL_NAME
    url   = f"{settings.HF_API_BASE}/{model}"

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            for attempt in range(HF_MAX_RETRY + 1):
                resp = await client.post(url, headers=_headers(), content=audio_bytes)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 503 and attempt < HF_MAX_RETRY:
                    wait = resp.json().get("estimated_time", 20)
                    logger.info(f"HF audio model loading, waiting {wait}s…")
                    await asyncio.sleep(min(float(wait), 30))
                    continue

                logger.error(f"HF API error {resp.status_code}: {resp.text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"HuggingFace API returned {resp.status_code}: {resp.text[:200]}"
                )

    except httpx.ConnectError as e:
        logger.error(f"HF API DNS/network error: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot reach HuggingFace API — DNS resolution failed. "
                "Ensure HF_TOKEN is set on your hosting platform and the server "
                f"has outbound internet access. Error: {e}"
            )
        )
    except httpx.TimeoutException as e:
        logger.error(f"HF API timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API timed out. The model may be cold-starting; please retry in 30 seconds."
        )

    raise HTTPException(status_code=502, detail="HF API: max retries exceeded.")


def parse_image_result(api_response: list[dict]) -> tuple[float, float]:
    """
    Parse HF image-classification response from capcheck/ai-image-detection.
    Labels: "Fake" / "Real"
    Returns (fake_prob, real_prob).
    """
    fake_prob = 0.5
    real_prob = 0.5

    for item in api_response:
        label = item.get("label", "").lower()
        score = float(item.get("score", 0.0))
        if any(k in label for k in ("fake", "deepfake", "manipulated", "artificial", "generated", "ai")):
            fake_prob = score
        elif any(k in label for k in ("real", "authentic", "genuine", "original")):
            real_prob = score

    # Normalize
    total = fake_prob + real_prob
    if total > 0:
        fake_prob /= total
        real_prob /= total

    return round(fake_prob, 4), round(real_prob, 4)


def parse_audio_result(api_response: list[dict]) -> tuple[float, float]:
    """
    Parse HF audio-classification response.
    Returns (fake_prob, real_prob).
    """
    return parse_image_result(api_response)  # same label parsing logic
