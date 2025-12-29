import argparse
import logging
import platform
from pathlib import Path
from .runner import BenchmarkRunner, BenchmarkConfig
from .config import OUTPUT_DIR, ensure_output_dir, TEST_TEXT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_reference_audio(output_path: Path):
    """Generate reference audio for ASR/VAD tests if it doesn't exist."""
    if output_path.exists():
        logger.info(f"Using existing reference audio: {output_path}")
        return

    logger.info("Generating reference audio for ASR/VAD benchmarks...")
    try:
        from src.voice_cloning.tts.cosyvoice import synthesize_speech
        synthesize_speech(
            "The quick brown fox jumps over the lazy dog. This is a reference audio for benchmarks.",
            output_path=str(output_path),
            use_mlx=(platform.system() == "Darwin" and platform.machine() == "arm64")
        )
        logger.info(f"Generated reference audio: {output_path}")
    except Exception as e:
        logger.warning(f"Failed to generate reference audio with CosyVoice: {e}")

def main():
    parser = argparse.ArgumentParser(description="Voice Cloning Benchmark Suite")
    parser.add_argument("--models", type=str, default="all",
                      help="Comma-separated list of models (e.g. 'whisper,marvis') or 'all'")
    parser.add_argument("--skip-asr", action="store_true", help="Skip ASR benchmarks")
    parser.add_argument("--skip-vad", action="store_true", help="Skip VAD benchmarks")
    parser.add_argument("--include-streaming", action="store_true", help="Include streaming tests")
    parser.add_argument("--include-cloning", action="store_true", help="Include cloning tests")
    parser.add_argument("--include-memory", action="store_true", default=True, help="Track memory usage")
    parser.add_argument("--no-memory", action="store_false", dest="include_memory", help="Disable memory tracking")
    
    args = parser.parse_args()
    models_to_run = args.models.split(",")
    
    def should_run(name):
        return "all" in models_to_run or name in models_to_run

    ensure_output_dir()
    reference_audio_path = OUTPUT_DIR / "benchmark_reference.wav"
    
    if not args.skip_asr or not args.skip_vad or args.include_cloning:
        generate_reference_audio(reference_audio_path)

    config = BenchmarkConfig(
        include_streaming=args.include_streaming,
        include_cloning=args.include_cloning,
        include_memory=args.include_memory,
        test_text=TEST_TEXT,
        reference_audio_path=str(reference_audio_path)
    )
    runner = BenchmarkRunner(config)
    
    # -------------------------------------------------------------------------
    # TTS Benchmarks
    # -------------------------------------------------------------------------
    if should_run("chatterbox"):
        try:
            from .tts.chatterbox import ChatterboxBenchmark
            runner.run_benchmark(ChatterboxBenchmark())
        except Exception as e:
            logger.warning(f"Skipping Chatterbox: {e}")

    if should_run("kitten"):
        try:
            from .tts.kitten import KittenBenchmark
            runner.run_benchmark(KittenBenchmark())
        except Exception as e:
            logger.warning(f"Skipping Kitten: {e}")

    if should_run("kokoro"):
        try:
            from .tts.kokoro import KokoroBenchmark
            runner.run_benchmark(KokoroBenchmark())
        except Exception as e:
            logger.warning(f"Skipping Kokoro: {e}")

    if should_run("marvis"):
        try:
            from .tts.marvis import MarvisBenchmark
            runner.run_benchmark(MarvisBenchmark())
        except Exception as e:
            logger.warning(f"Skipping Marvis: {e}")

    if should_run("supertone"):
        try:
            from .tts.supertone import SupertoneBenchmark
            runner.run_benchmark(SupertoneBenchmark())
        except Exception as e:
            logger.warning(f"Skipping Supertone: {e}")

    if should_run("neutts"):
        try:
            from .tts.neutts import NeuTTSBenchmark
            runner.run_benchmark(NeuTTSBenchmark(ref_audio_path=str(reference_audio_path)))
        except Exception as e:
            logger.warning(f"Skipping NeuTTS: {e}")

    if should_run("cosyvoice"):
        try:
            from .tts.cosyvoice import CosyVoiceBenchmark
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                runner.run_benchmark(CosyVoiceBenchmark(use_mlx=True))
            runner.run_benchmark(CosyVoiceBenchmark(use_mlx=False))
        except Exception as e:
            logger.warning(f"Skipping CosyVoice: {e}")

    # -------------------------------------------------------------------------
    # ASR Benchmarks
    # -------------------------------------------------------------------------
    if not args.skip_asr:
        # Whisper Variants
        if should_run("whisper") or "all" in models_to_run:
            try:
                from .asr.whisper import WhisperBenchmark
                # Run various sizes 
                # MLX 
                if platform.system() == "Darwin" and platform.machine() == "arm64":
                    runner.run_benchmark(WhisperBenchmark(use_mlx=True, model_id="mlx-community/whisper-large-v3-turbo"))
                    runner.run_benchmark(WhisperBenchmark(use_mlx=True, model_id="mlx-community/whisper-tiny-mlx"))
                    runner.run_benchmark(WhisperBenchmark(use_mlx=True, model_id="mlx-community/whisper-base-mlx"))
                    # Skip massive ones unless specifically asked? User said "discriminate properly" 
                    # We will stick to efficient ones for 'all' or explicit if requested
                else:
                    # PyTorch
                    runner.run_benchmark(WhisperBenchmark(use_mlx=False, model_id="openai/whisper-tiny"))
                    runner.run_benchmark(WhisperBenchmark(use_mlx=False, model_id="openai/whisper-base"))
            except Exception as e:
                logger.warning(f"Skipping Whisper: {e}")

        # Canary
        if should_run("canary") or "all" in models_to_run:
            try:
                from .asr.canary import CanaryBenchmark
                runner.run_benchmark(CanaryBenchmark())
            except Exception as e:
                logger.warning(f"Skipping Canary: {e}")
        
        # Parakeet
        if should_run("parakeet") or "all" in models_to_run:
            try:
                from .asr.parakeet import ParakeetBenchmark
                runner.run_benchmark(ParakeetBenchmark())
            except Exception as e:
                logger.warning(f"Skipping Parakeet: {e}")

    # -------------------------------------------------------------------------
    # VAD Benchmarks
    # -------------------------------------------------------------------------
    if not args.skip_vad and (should_run("humaware") or "all" in models_to_run):
        try:
            from .vad.humaware import HumAwareBenchmark
            runner.run_benchmark(HumAwareBenchmark())
        except Exception as e:
             logger.warning(f"Skipping HumAware: {e}")
    
    runner.save_report()

if __name__ == "__main__":
    main()
