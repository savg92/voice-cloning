import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

class FeatureValidator:
    def __init__(self, output_dir: str = "outputs/verification"):
        # Use absolute path to be certain
        self.output_dir = Path(output_dir).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary = []

    def log_result(self, feature: str, success: bool, message: str):
        status = "✅" if success else "❌"
        self.summary.append({"feature": feature, "status": status, "message": message})
        logger.info(f"{status} {feature}: {message}")

    def test_kokoro_pytorch(self):
        logger.info("Testing Kokoro (PyTorch)...")
        try:
            from src.voice_cloning.tts.kokoro import synthesize_speech
            out = self.output_dir / "kokoro_pytorch.wav"
            out_str = str(out)
            
            # Clean up
            if out.exists():
                out.unlink()
            
            result_path = synthesize_speech("Hello from PyTorch.", output_path=out_str, use_mlx=False)
            logger.info(f"Synthesize speech returned path: {result_path}")
            
            if out.exists():
                self.log_result("Kokoro PyTorch", True, f"Generated {out_str}")
            else:
                self.log_result("Kokoro PyTorch", False, f"File not generated at {out_str}")
        except Exception as e:
            self.log_result("Kokoro PyTorch", False, str(e))

    def test_kokoro_mlx(self):
        logger.info("Testing Kokoro (MLX)...")
        if sys.platform != "darwin":
            self.log_result("Kokoro MLX", True, "Skipped (non-macOS)")
            return
            
        try:
            from src.voice_cloning.tts.kokoro import synthesize_speech
            
            # Test English
            out_en = self.output_dir / "kokoro_mlx_en.wav"
            out_en_str = str(out_en)
            if out_en.exists():
                out_en.unlink()
            synthesize_speech("Hello from MLX English.", output_path=out_en_str, use_mlx=True, lang_code='a')
            
            # Test Spanish
            out_es = self.output_dir / "kokoro_mlx_es.wav"
            out_es_str = str(out_es)
            if out_es.exists():
                out_es.unlink()
            synthesize_speech("Hola desde MLX español.", output_path=out_es_str, use_mlx=True, lang_code='e')
            
            success = True
            missing = []
            
            if not out_en.exists():
                success = False
                missing.append(out_en_str)
            
            if not out_es.exists():
                success = False
                missing.append(out_es_str)

            if success:
                self.log_result("Kokoro MLX", True, "English and Spanish files generated")
            else:
                self.log_result("Kokoro MLX", False, f"Missing files: {', '.join(missing)}")
        except Exception as e:
            self.log_result("Kokoro MLX", False, str(e))

    def test_asr_whisper(self):
        logger.info("Testing Whisper ASR...")
        try:
            from src.voice_cloning.asr.whisper import WhisperASR
            model = WhisperASR(model_id="openai/whisper-tiny") # Fast test
            model.load_model()
            
            audio = Path("samples/anger.wav").absolute()
            if not audio.exists():
                self.log_result("Whisper ASR", False, f"Input audio {audio} missing")
                return

            result = model.transcribe(str(audio))
            text = result if isinstance(result, str) else result.get("text", "")
            
            if len(text) > 0:
                self.log_result("Whisper ASR", True, f"Transcript: {text[:30]}...")
            else:
                self.log_result("Whisper ASR", False, "Empty transcript")
        except Exception as e:
            self.log_result("Whisper ASR", False, str(e))

    def run_all(self):
        logger.info("Starting Comprehensive Feature Verification")
        logger.info("=" * 40)
        
        self.test_kokoro_pytorch()
        self.test_kokoro_mlx()
        self.test_asr_whisper()
        
        logger.info("\nVerification Summary:")
        logger.info("-" * 40)
        for item in self.summary:
            logger.info(f"{item['status']} {item['feature']}: {item['message']}")
        logger.info("-" * 40)

if __name__ == "__main__":
    validator = FeatureValidator()
    validator.run_all()
