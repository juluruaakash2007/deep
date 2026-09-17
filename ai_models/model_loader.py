"""
DeepShield — Model Loader
In HF API mode, no local models are loaded.
This module is kept for compatibility but is a no-op.
"""

import logging
logger = logging.getLogger(__name__)

_models: dict = {}


def get_model(name: str):
    return _models.get(name)


def set_model(name: str, model):
    _models[name] = model


def get_device():
    return "cpu"


async def load_all_models():
    """No-op in HF API mode — models run on HuggingFace servers."""
    logger.info("HF API mode: no local models to load.")
