import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_streaming():
    from src.voice_cloning.tts.kokoro import synthesize_speech
    
    text = "This is a streaming test. It should play audio chunks as they are generated."
    output_dir = Path("outputs/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test PyTorch Streaming
    logger.info("=== Testing PyTorch Streaming ===")
    try:
        out_pytorch = output_dir / "streaming_pytorch.wav"
        synthesize_speech(text, output_path=str(out_pytorch), use_mlx=False, stream=True)
        logger.info(f"PyTorch streaming finished. Output: {out_pytorch}")
    except Exception as e:
        logger.error(f"PyTorch streaming failed: {e}")

    # Test MLX Streaming
    if sys.platform == "darwin":
        logger.info("\n=== Testing MLX Streaming ===")
        try:
            out_mlx = output_dir / "streaming_mlx.wav"
            synthesize_speech(text, output_path=str(out_mlx), use_mlx=True, stream=True)
            logger.info(f"MLX streaming finished. Output: {out_mlx}")
        except Exception as e:
            logger.error(f"MLX streaming failed: {e}")

if __name__ == "__main__":
    test_streaming()
