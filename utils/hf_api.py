"""
DeepShield — HuggingFace Inference API Client
==============================================
Image model : capcheck/ai-image-detection  (ViT-Base, CIFAKE-trained)
Audio model : HamedAGH/deepfake-audio-detection

Retry strategy:
  - 503 from HF (model loading)  → wait estimated_time, retry up to HF_MAX_RETRY times
  - ConnectError / DNS failure    → exponential backoff, retry up to DNS_MAX_RETRY times
  - TimeoutException              → raise 504 immediately (no retry — already waited 60s)
"""

import asyncio
import logging
import socket
import httpx
from fastapi import HTTPException
from backend.config import settings

logger = logging.getLogger(__name__)

HF_HOST        = "router.huggingface.co"  # new endpoint (api-inference.huggingface.co is decommissioned)
HF_TIMEOUT     = 60.0   # seconds — HF cold-start can take ~20s
HF_MAX_RETRY   = 2       # retries on HF 503 (model loading)
DNS_MAX_RETRY  = 3       # retries on ConnectError / DNS failure
DNS_RETRY_BASE = 2.0     # exponential backoff base (2s, 4s, 8s)


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.HF_TOKEN}"}


def _check_dns() -> bool:
    """
    Fast synchronous DNS pre-check using socket.
    Returns True if hostname resolves, False otherwise.
    Logs the resolved IP so Render logs can confirm network access.
    """
    try:
        addrs = socket.getaddrinfo(HF_HOST, 443, proto=socket.IPPROTO_TCP)
        if addrs:
            ip = addrs[0][4][0]
            logger.info(f"DNS OK — {HF_HOST} → {ip}")
            return True
        return False
    except socket.gaierror as e:
        logger.warning(f"DNS check failed for {HF_HOST}: {e}")
        return False


async def _call_with_dns_retry(
    client: httpx.AsyncClient,
    url: str,
    content: bytes,
) -> httpx.Response:
    """
    POST `content` to `url`, retrying on ConnectError with exponential backoff.
    Raises HTTPException(503) if all DNS_MAX_RETRY attempts fail.
    """
    last_err: Exception | None = None

    for dns_attempt in range(DNS_MAX_RETRY + 1):
        if dns_attempt > 0:
            wait = DNS_RETRY_BASE ** dns_attempt
            logger.warning(f"ConnectError — DNS retry {dns_attempt}/{DNS_MAX_RETRY} in {wait:.0f}s…")
            await asyncio.sleep(wait)

        try:
            return await client.post(url, headers=_headers(), content=content)
        except httpx.ConnectError as e:
            last_err = e
            logger.error(f"HF ConnectError (attempt {dns_attempt + 1}): {e}")
            continue

    raise HTTPException(
        status_code=503,
        detail=(
            f"Cannot reach HuggingFace API after {DNS_MAX_RETRY + 1} attempts — "
            "DNS resolution failed. Ensure HF_TOKEN is set on Render and the server "
            f"has outbound internet access. Last error: {last_err}"
        ),
    )


async def hf_image_classify(image_bytes: bytes, model: str | None = None) -> list[dict]:
    """
    POST raw image bytes to HF image-classification endpoint.
    Returns: [{"label": "Fake", "score": 0.95}, {"label": "Real", "score": 0.05}]
    """
    if not settings.HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not configured. Set it as an environment variable on Render.",
        )

    model = model or settings.IMAGE_MODEL_NAME
    url   = f"{settings.HF_API_BASE}/{model}"
    logger.info(f"HF image classify → {url}")

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            for attempt in range(HF_MAX_RETRY + 1):
                resp = await _call_with_dns_retry(client, url, image_bytes)

                if resp.status_code == 200:
                    return resp.json()

                # 503 = HF model still loading — wait and retry
                if resp.status_code == 503 and attempt < HF_MAX_RETRY:
                    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    wait = body.get("estimated_time", 20)
                    logger.info(f"HF model loading, waiting {wait}s…")
                    await asyncio.sleep(min(float(wait), 30))
                    continue

                logger.error(f"HF API error {resp.status_code}: {resp.text[:300]}")
                raise HTTPException(
                    status_code=502,
                    detail=f"HuggingFace API returned {resp.status_code}: {resp.text[:200]}",
                )

    except HTTPException:
        raise  # re-raise our own clean errors
    except httpx.TimeoutException as e:
        logger.error(f"HF API timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API timed out (60s). Model may be cold-starting — retry in 30 seconds.",
        )

    raise HTTPException(status_code=502, detail="HF API: max retries exceeded.")


async def hf_audio_classify(audio_bytes: bytes, model: str | None = None) -> list[dict]:
    """
    POST raw audio bytes to HF audio-classification endpoint.
    Returns: [{"label": "fake", "score": 0.88}, {"label": "real", "score": 0.12}]
    """
    if not settings.HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN is not configured. Set it as an environment variable on Render.",
        )

    model = model or settings.AUDIO_MODEL_NAME
    url   = f"{settings.HF_API_BASE}/{model}"
    logger.info(f"HF audio classify → {url}")

    try:
        async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
            for attempt in range(HF_MAX_RETRY + 1):
                resp = await _call_with_dns_retry(client, url, audio_bytes)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 503 and attempt < HF_MAX_RETRY:
                    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    wait = body.get("estimated_time", 20)
                    logger.info(f"HF audio model loading, waiting {wait}s…")
                    await asyncio.sleep(min(float(wait), 30))
                    continue

                logger.error(f"HF API error {resp.status_code}: {resp.text[:300]}")
                raise HTTPException(
                    status_code=502,
                    detail=f"HuggingFace API returned {resp.status_code}: {resp.text[:200]}",
                )

    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        logger.error(f"HF API timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API timed out (60s). Model may be cold-starting — retry in 30 seconds.",
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

    total = fake_prob + real_prob
    if total > 0:
        fake_prob /= total
        real_prob /= total

    return round(fake_prob, 4), round(real_prob, 4)


def parse_audio_result(api_response: list[dict]) -> tuple[float, float]:
    """Parse HF audio-classification response. Returns (fake_prob, real_prob)."""
    return parse_image_result(api_response)
