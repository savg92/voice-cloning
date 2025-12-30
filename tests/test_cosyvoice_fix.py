import logging
import os
from voice_cloning.tts.cosyvoice import synthesize_speech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cosyvoice():
    text = "This is a test of the CosyVoice synthesis system."
    output_dir = "tests/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # We need a reference audio for zero-shot
    ref_audio = "samples/anger.wav" # Using an existing sample
    if not os.path.exists(ref_audio):
         # Create a dummy or find another
         ref_audio = None
         logger.warning("No reference audio found at samples/anger.wav. Zero-shot might fail if no default injected.")

    try:
        logger.info("Testing CosyVoice with MLX backend...")
        out_mlx = os.path.join(output_dir, "test_cosy_mlx.wav")
        synthesize_speech(text, out_mlx, use_mlx=True, ref_audio_path=ref_audio)
        logger.info(f"MLX Success: {out_mlx}")
    except Exception as e:
        logger.error(f"MLX Failed: {e}")

    try:
        logger.info("Testing CosyVoice with PyTorch backend...")
        out_pt = os.path.join(output_dir, "test_cosy_pt.wav")
        synthesize_speech(text, out_pt, use_mlx=False, ref_audio_path=ref_audio, ref_text="Be careful.")
        logger.info(f"PyTorch Success: {out_pt}")
    except Exception as e:
        logger.error(f"PyTorch Failed: {e}")

if __name__ == "__main__":
    test_cosyvoice()
