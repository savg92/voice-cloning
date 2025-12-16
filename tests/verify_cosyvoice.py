#!/usr/bin/env python3
"""
Verification script for CosyVoice2 implementation.
Tests both MLX and PyTorch backends if available.
"""

import os
import sys
from pathlib import Path
import logging
import soundfile as sf
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add local CosyVoice repo to path if it exists
REPOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COSYVOICE_PATH = os.path.join(REPOS_ROOT, "models", "CosyVoice")
if os.path.exists(COSYVOICE_PATH):
    if COSYVOICE_PATH not in sys.path:
        sys.path.append(COSYVOICE_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CosyVoiceVerifier")

def create_dummy_reference(path, duration=3.0, sr=16000):
    """Create a dummy reference audio file."""
    # Create a sine wave as dummy reference
    t = np.linspace(0, duration, int(sr * duration), False)
    # A simple chord
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    sf.write(path, audio, sr)
    return path

def verify_mlx():
    logger.info("="*40)
    logger.info("Verifying MLX Backend (CosyVoice2)")
    logger.info("="*40)
    
    try:
        from voice_cloning.tts.cosyvoice import synthesize_speech
        
        output_dir = Path("outputs/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ref_path = output_dir / "ref.wav"
        if not ref_path.exists():
            create_dummy_reference(str(ref_path))
        
        # 1. Basic Synthesis
        logger.info("1. Testing Basic Synthesis (using dummy ref)...")
        out_basic = output_dir / "mlx_basic.wav"
        synthesize_speech(
            "Hello, this is a test of CosyVoice2 on MLX.",
            output_path=str(out_basic),
            ref_audio_path=str(ref_path),
            use_mlx=True
        )
        if out_basic.exists():
            logger.info(f"✓ Basic synthesis successful: {out_basic}")
        else:
            logger.error("✗ Basic synthesis failed to produce file")
            
        # 2. Zero-shot Cloning
        logger.info("2. Testing Zero-shot Cloning...")
        out_clone = output_dir / "mlx_clone.wav"
        synthesize_speech(
            "This is a zero-shot voice cloning test.",
            output_path=str(out_clone),
            ref_audio_path=str(ref_path),
            use_mlx=True
        )
        if out_clone.exists():
            logger.info(f"✓ Zero-shot cloning successful: {out_clone}")
            
    except ImportError as e:
        logger.error(f"Skipping MLX verification: {e}")
    except Exception as e:
        logger.error(f"MLX verification failed: {e}")

def verify_pytorch():
    logger.info("\n" + "="*40)
    logger.info("Verifying PyTorch Backend (CosyVoice2)")
    logger.info("="*40)
    
    try:
        from voice_cloning.tts.cosyvoice import synthesize_speech
        # Try to import cosyvoice to check availability
        try:
             import cosyvoice
             logger.info("CosyVoice package found.")
        except ImportError:
             logger.warning("CosyVoice package not found in path. Skipping PyTorch verification.")
             return

        output_dir = Path("outputs/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ref_path = output_dir / "ref.wav"
        if not ref_path.exists():
            create_dummy_reference(str(ref_path))
            
        # 1. Basic Synthesis
        logger.info("1. Testing Basic Synthesis (SFT)...")
        out_basic = output_dir / "torch_basic.wav"
        try:
            synthesize_speech(
                "Hello, this is a test of CosyVoice2 on PyTorch.",
                output_path=str(out_basic),
                ref_audio_path=str(ref_path), # Provide ref audio
                use_mlx=False
            )
            if out_basic.exists():
                logger.info(f"✓ Basic synthesis successful: {out_basic}")
        except Exception as e:
             logger.error(f"Synthesis failed: {e}")
             # Print stack trace for debugging if needed
             import traceback
             traceback.print_exc()

    except Exception as e:
        logger.error(f"PyTorch verification Setup failed: {e}")

if __name__ == "__main__":
    verify_mlx()
    verify_pytorch()
