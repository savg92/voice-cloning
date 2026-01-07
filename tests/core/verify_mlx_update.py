
import os
import torch
import numpy as np
import soundfile as sf
import logging
import pytest
from src.voice_cloning.tts.kokoro import KokoroWrapper
from src.voice_cloning.tts.marvis import MarvisWrapper
from src.voice_cloning.tts.chatterbox import ChatterboxWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_kokoro_mlx():
    """Verify Kokoro MLX backend still works."""
    try:
        logger.info("Testing Kokoro MLX...")
        kokoro = KokoroWrapper(use_mlx=True)
        text = "Hello, this is a test of the MLX update for Kokoro."
        audio, sr = kokoro.generate(text, voice="af_heart")
        assert audio is not None
        assert len(audio) > 0
        logger.info("✓ Kokoro MLX success")
    except Exception as e:
        logger.error(f"Kokoro MLX failed: {e}")
        raise

def test_marvis_mlx():
    """Verify Marvis MLX backend still works (requires patched sesame.py)."""
    try:
        logger.info("Testing Marvis MLX...")
        marvis = MarvisWrapper(use_mlx=True)
        text = "Hello, this is a test of the MLX update for Marvis."
        audio, sr = marvis.generate(text)
        assert audio is not None
        assert len(audio) > 0
        logger.info("✓ Marvis MLX success")
    except Exception as e:
        logger.error(f"Marvis MLX failed: {e}")
        raise

def test_chatterbox_mlx():
    """Verify Chatterbox MLX backend still works."""
    try:
        logger.info("Testing Chatterbox MLX...")
        chatterbox = ChatterboxWrapper()
        text = "Hello, this is a test of the MLX update for Chatterbox."
        # Use a short text and default ref_audio if possible
        audio, sr = chatterbox.generate(text, use_mlx=True)
        assert audio is not None
        assert len(audio) > 0
        logger.info("✓ Chatterbox MLX success")
    except Exception as e:
        logger.error(f"Chatterbox MLX failed: {e}")
        raise

if __name__ == "__main__":
    # Run tests manually
    test_kokoro_mlx()
    test_marvis_mlx()
    test_chatterbox_mlx()
