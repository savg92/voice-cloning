import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

def test_multilang():
    from voice_cloning.tts.kokoro import synthesize_speech
    
    # Language map based on user report + standard Kokoro codes
    # German (d), Chinese (z), Japanese (j), Russian (r), Turkish (t)
    test_cases = [
        ("d", "Hallo Welt", "de_test.wav"),
        ("z", "你好世界", "zh_test.wav"),
        ("j", "こんにちは世界", "ja_test.wav"),
        ("r", "Привет мир", "ru_test.wav"),
        ("t", "Merhaba Dünya", "tr_test.wav"),
    ]
    
    output_dir = Path("outputs/repro")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lang_code, text, filename in test_cases:
        logger.info(f"--- Testing Language: {lang_code} ---")
        
        # Test MLX
        if sys.platform == "darwin":
            try:
                out_path = output_dir / f"mlx_{filename}"
                if out_path.exists():
                    out_path.unlink()
                
                logger.info(f"Generating MLX for {lang_code}...")
                synthesize_speech(text, output_path=str(out_path), lang_code=lang_code, use_mlx=True)
                
                if out_path.exists():
                    logger.info(f"✅ MLX {lang_code} Success")
                else:
                    logger.error(f"❌ MLX {lang_code} Failed (No file)")
            except Exception as e:
                logger.error(f"❌ MLX {lang_code} Error: {e}")

        # Test PyTorch
        try:
            out_path = output_dir / f"torch_{filename}"
            if out_path.exists():
                out_path.unlink()
            
            logger.info(f"Generating PyTorch for {lang_code}...")
            synthesize_speech(text, output_path=str(out_path), lang_code=lang_code, use_mlx=False)
            
            if out_path.exists():
                logger.info(f"✅ PyTorch {lang_code} Success")
            else:
                logger.error(f"❌ PyTorch {lang_code} Failed (No file)")
        except Exception as e:
            logger.error(f"❌ PyTorch {lang_code} Error: {e}")

if __name__ == "__main__":
    test_multilang()
