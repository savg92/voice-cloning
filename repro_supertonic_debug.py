import os
import logging
import soundfile as sf
import numpy as np
from voice_cloning.tts.supertonic2 import Supertonic2TTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config(name, providers=None, steps=10):
    logger.info(f"Testing config: {name} (Steps: {steps}, Providers: {providers})")
    
    # We need to hack the class or re-instantiate to change providers since it's done in __init__
    # So we will instantiate a new one.
    # To pass providers, we need to modify the class or subclass it.
    # For this script, I'll just rely on the fact that I can't easily change providers without code change
    # UNLESS I mock onnxruntime or modify the source.
    
    # Let's modify the source file to allow provider configuration first.
    pass

if __name__ == "__main__":
    # This script is just a placeholder until I modify the source code
    pass
