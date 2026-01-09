import os
import torch
import numpy as np
import soundfile as sf
import logging
from src.voice_cloning.tts.kokoro import synthesize_speech as kokoro_synthesize
from src.voice_cloning.tts.marvis import MarvisTTS
from src.voice_cloning.tts.chatterbox import _synthesize_with_mlx as chatterbox_mlx_synthesize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_kokoro_mlx():
    """Verify Kokoro MLX backend still works."""
    try:
        logger.info("Testing Kokoro MLX...")
        output_path = "tests/output/verify_kokoro_mlx.wav"
        os.makedirs("tests/output", exist_ok=True)
        kokoro_synthesize("Hello, this is a test of the MLX update for Kokoro.", 
                           output_path=output_path, use_mlx=True)
        assert os.path.exists(output_path)
        logger.info("✓ Kokoro MLX success")
    except Exception as e:
        logger.error(f"Kokoro MLX failed: {e}")
        raise

def test_marvis_mlx():
    """Verify Marvis MLX backend still works."""
    try:
        logger.info("Testing Marvis MLX...")
        marvis = MarvisTTS()
        output_path = "tests/output/verify_marvis_mlx.wav"
        os.makedirs("tests/output", exist_ok=True)
        marvis.synthesize("Hello, this is a test of the MLX update for Marvis.", 
                          output_path=output_path)
        assert os.path.exists(output_path)
        logger.info("✓ Marvis MLX success")
    except Exception as e:
        logger.error(f"Marvis MLX failed: {e}")
        raise

def test_chatterbox_mlx():
    """Verify Chatterbox MLX backend still works."""
    try:
        logger.info("Testing Chatterbox MLX...")
        output_path = "tests/output/verify_chatterbox_mlx.wav"
        os.makedirs("tests/output", exist_ok=True)
        # Chatterbox MLX usually needs a reference audio for voice cloning in some versions
        # Let's use one from samples
        ref_audio = "samples/kokoro_voices/af_heart.wav"
        chatterbox_mlx_synthesize("Hello, this is a test of the MLX update for Chatterbox.", 
                                   output_wav=output_path, source_wav=ref_audio)
        assert os.path.exists(output_path)
        logger.info("✓ Chatterbox MLX success")
    except Exception as e:
        logger.error(f"Chatterbox MLX failed: {e}")
        raise

if __name__ == "__main__":
    test_kokoro_mlx()
    test_marvis_mlx()
    test_chatterbox_mlx()