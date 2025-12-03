#!/usr/bin/env python3
"""
Benchmark Suite for Voice Cloning Models
Evaluates performance (Latency, RTF) of TTS, ASR, and VAD models.
"""

import time
import sys
import os
import argparse
import logging
import numpy as np
import soundfile as sf
import torch
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TEST_TEXT = "The quick brown fox jumps over the lazy dog. This is a benchmark test to measure synthesis speed."
OUTPUT_DIR = Path("outputs/benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = "BENCHMARK_RESULTS.md"

@dataclass
class BenchmarkResult:
    model_name: str
    task_type: str
    latency_ms: float
    rtf: float
    audio_duration_s: float
    memory_mb: float = 0.0
    ttfa_ms: float = 0.0  # Time to first audio (for streaming)
    notes: str = ""
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

class BenchmarkRunner:
    def __init__(self, device: Optional[str] = None, include_streaming: bool = False, 
                 include_cloning: bool = False, include_memory: bool = True):
        self.device = device
        self.include_streaming = include_streaming
        self.include_cloning = include_cloning
        self.include_memory = include_memory
        self.results: List[BenchmarkResult] = []
        self.test_audio_path: Optional[Path] = None
        self.reference_audio_path: Optional[Path] = None

    def _get_device(self) -> str:
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _measure_streaming_ttfa(self, model_name: str, model, text: str) -> Tuple[float, List[np.ndarray]]:
        """
        Measure Time-To-First-Audio for streaming models.
        Returns (ttfa_ms, chunks) where chunks is list of audio arrays.
        """
        chunks = []
        ttfa = None
        start_time = time.time()
        
        if model_name == "Kokoro":
            from src.voice_cloning.tts.kokoro import synthesize_speech
            # Kokoro uses generator internally, measure time to first chunk
            # For simplicity, we'll measure total time as it already streams internally
            ttfa = 0  # Kokoro doesn't expose TTFA easily
        elif model_name == "Marvis (MLX)":
            # Marvis has streaming support
            try:
                for i, chunk in enumerate(model.synthesize_stream(text)):
                    if ttfa is None:
                        ttfa = (time.time() - start_time) * 1000
                    chunks.append(chunk)
            except AttributeError:
                logger.warning(f"  {model_name} doesn't support streaming")
                return 0, []
        
        if ttfa is None:
            ttfa = 0
            
        return ttfa, chunks

    def generate_test_audio(self):
        """Generates a test audio file using Kokoro TTS for ASR/VAD benchmarking."""
        logger.info("Generating test audio for ASR/VAD benchmarks...")
        try:
            from src.voice_cloning.tts.kokoro import synthesize_speech
            self.test_audio_path = OUTPUT_DIR / "benchmark_reference.wav"
            synthesize_speech(TEST_TEXT, output_path=str(self.test_audio_path))
            logger.info(f"Test audio generated at {self.test_audio_path}")
            
            # Save transcript
            self.test_text_path = OUTPUT_DIR / "benchmark_reference.txt"
            with open(self.test_text_path, "w") as f:
                f.write(TEST_TEXT)
            
            # Also use as reference audio for voice cloning (copy with different name)
            if self.include_cloning:
                self.reference_audio_path = OUTPUT_DIR / "cloning_reference.wav"
                self.reference_text_path = OUTPUT_DIR / "cloning_reference.txt"
                import shutil
                shutil.copy(self.test_audio_path, self.reference_audio_path)
                shutil.copy(self.test_text_path, self.reference_text_path)
                logger.info(f"Reference audio for cloning: {self.reference_audio_path}")
        except Exception as e:
            logger.error(f"Failed to generate test audio: {e}")
            # Create dummy audio if TTS fails
            self.test_audio_path = OUTPUT_DIR / "benchmark_dummy.wav"
            dummy_audio = np.random.uniform(-0.5, 0.5, 24000 * 5) # 5 seconds noise
            sf.write(str(self.test_audio_path), dummy_audio, 24000)

    def benchmark_tts(self, model_name: str, wrapper_class, **kwargs):
        logger.info(f"Benchmarking TTS: {model_name}...")
        try:
            # Initialize
            start_init = time.time()
            
            # Handle different init signatures
            if model_name == "KittenTTS (Nano)":
                model = wrapper_class("KittenML/kitten-tts-nano-0.2")
            elif model_name == "Supertone":
                model = wrapper_class()
            elif model_name == "Kokoro":
                # Kokoro doesn't have a class, use function directly
                model = None
            elif model_name == "NeuTTS Air":
                if not self.reference_audio_path or not self.reference_audio_path.exists():
                    logger.warning("  Skipping: NeuTTS Air requires reference audio (enable --include-cloning)")
                    return
                model = wrapper_class(backbone_device=self._get_device(), codec_device=self._get_device())
            else:
                model = wrapper_class(device=self._get_device(), **kwargs)
                
            init_time = time.time() - start_init
            logger.info(f"  Initialization time: {init_time:.2f}s")

            # Warmup
            logger.info("  Warming up...")
            if model_name == "Kokoro":
                from src.voice_cloning.tts.kokoro import synthesize_speech
                synthesize_speech("Warmup", output_path=str(OUTPUT_DIR / "warmup_kokoro.wav"))
            elif model_name == "Marvis (MLX)":
                model.synthesize("Warmup", output_path=str(OUTPUT_DIR / "warmup.wav"))
            elif model_name == "Supertone":
                model.synthesize("Warmup", output_path=str(OUTPUT_DIR / "warmup_supertone.wav"))
            elif model_name == "KittenTTS (Nano)":
                model.synthesize_to_file("Warmup", str(OUTPUT_DIR / "warmup_kitten.wav"))
            elif model_name == "NeuTTS Air":
                model.synthesize(
                    text="Warmup",
                    output_path=str(OUTPUT_DIR / "warmup_neutts.wav"),
                    ref_audio_path=str(self.reference_audio_path),
                    ref_text_path=str(self.reference_text_path)
                )
            else:
                model.synthesize("Warmup")

            # Benchmark
            logger.info("  Running benchmark...")
            
            # Memory tracking
            mem_before = self._get_memory_usage() if self.include_memory else 0
            gc.collect()
            
            start_time = time.time()
            
            if model_name == "Kokoro":
                from src.voice_cloning.tts.kokoro import synthesize_speech
                out_path = str(OUTPUT_DIR / "bench_kokoro.wav")
                synthesize_speech(TEST_TEXT, output_path=out_path)
                audio, sr = sf.read(out_path)
            elif model_name == "Marvis (MLX)":
                audio = model.synthesize(TEST_TEXT, output_path=str(OUTPUT_DIR / "bench_marvis.wav"))
                # Marvis returns path, load it to get duration
                audio, sr = sf.read(str(OUTPUT_DIR / "bench_marvis.wav"))
            elif model_name == "Supertone":
                out_path = str(OUTPUT_DIR / "bench_supertone.wav")
                model.synthesize(TEST_TEXT, output_path=out_path)
                audio, sr = sf.read(out_path)
            elif model_name == "KittenTTS (Nano)":
                out_path = str(OUTPUT_DIR / "bench_kitten.wav")
                model.synthesize_to_file(TEST_TEXT, out_path)
                audio, sr = sf.read(out_path)
            elif model_name == "NeuTTS Air":
                out_path = str(OUTPUT_DIR / "bench_neutts.wav")
                model.synthesize(
                    text=TEST_TEXT,
                    output_path=out_path,
                    ref_audio_path=str(self.reference_audio_path),
                    ref_text_path=str(self.reference_text_path)
                )
                audio, sr = sf.read(out_path)
            else:
                audio = model.synthesize(TEST_TEXT)

                
            end_time = time.time()
            
            # Memory tracking
            mem_after = self._get_memory_usage() if self.include_memory else 0
            memory_used = max(0, mem_after - mem_before)
            
            latency = end_time - start_time
            
            # Calculate duration
            if isinstance(audio, tuple): # Some return (sr, audio)
                sr, audio = audio
            elif not 'sr' in locals():
                sr = 24000 # Default assumption
                if model_name == "KittenTTS (Nano)": sr = 16000 
            
            duration = len(audio) / sr
            rtf = latency / duration
            
            result = BenchmarkResult(
                model_name=model_name,
                task_type="TTS",
                latency_ms=latency * 1000,
                rtf=rtf,
                audio_duration_s=duration,
                memory_mb=memory_used,
                notes=f"Device: {self._get_device()}"
            )
            self.results.append(result)
            logger.info(f"  Result: Latency={latency*1000:.2f}ms, RTF={rtf:.4f}")
            
        except Exception as e:
            logger.error(f"  Failed to benchmark {model_name}: {e}")
            self.results.append(BenchmarkResult(model_name, "TTS", 0, 0, 0, f"FAILED: {str(e)}"))

    def benchmark_asr(self, model_name: str, wrapper_class, **kwargs):
        if not self.test_audio_path or not self.test_audio_path.exists():
            logger.warning("Skipping ASR benchmark: Test audio not available")
            return

        logger.info(f"Benchmarking ASR: {model_name}...")
        try:
            # Initialize
            if model_name == "Canary":
                model = wrapper_class()
            else:
                model = wrapper_class(device=self._get_device(), **kwargs)
            
            # Warmup
            try:
                model.transcribe(str(self.test_audio_path))
            except:
                pass

            # Benchmark
            start_time = time.time()
            model.transcribe(str(self.test_audio_path))
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Get audio duration
            info = sf.info(str(self.test_audio_path))
            duration = info.duration
            rtf = latency / duration
            
            result = BenchmarkResult(
                model_name=model_name,
                task_type="ASR",
                latency_ms=latency * 1000,
                rtf=rtf,
                audio_duration_s=duration,
                notes=f"Device: {self._get_device()}"
            )
            self.results.append(result)
            logger.info(f"  Result: Latency={latency*1000:.2f}ms, RTF={rtf:.4f}")

        except Exception as e:
            logger.error(f"  Failed to benchmark {model_name}: {e}")
            self.results.append(BenchmarkResult(model_name, "ASR", 0, 0, 0, f"FAILED: {str(e)}"))

    def benchmark_vad(self, model_name: str, wrapper_class, **kwargs):
        if not self.test_audio_path or not self.test_audio_path.exists():
            logger.warning("Skipping VAD benchmark: Test audio not available")
            return

        logger.info(f"Benchmarking VAD: {model_name}...")
        try:
            # Initialize
            model = wrapper_class(**kwargs) # VAD might not take device arg in init
            
            # Benchmark
            start_time = time.time()
            model.detect_speech(str(self.test_audio_path))
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Get audio duration
            info = sf.info(str(self.test_audio_path))
            duration = info.duration
            rtf = latency / duration
            
            result = BenchmarkResult(
                model_name=model_name,
                task_type="VAD",
                latency_ms=latency * 1000,
                rtf=rtf,
                audio_duration_s=duration,
                notes="CPU" # VAD usually runs on CPU
            )
            self.results.append(result)
            logger.info(f"  Result: Latency={latency*1000:.2f}ms, RTF={rtf:.4f}")

        except Exception as e:
            logger.error(f"  Failed to benchmark {model_name}: {e}")
            self.results.append(BenchmarkResult(model_name, "VAD", 0, 0, 0, f"FAILED: {str(e)}"))

    def save_report(self):
        logger.info(f"Saving report to {REPORT_FILE}...")
        with open(REPORT_FILE, "w") as f:
            f.write("# Voice Cloning Benchmark Results\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Device**: {self._get_device().upper()}\n")
            f.write(f"**Streaming Tests**: {'Enabled' if self.include_streaming else 'Disabled'}\n")
            f.write(f"**Cloning Tests**: {'Enabled' if self.include_cloning else 'Disabled'}\n")
            f.write(f"**Memory Tracking**: {'Enabled' if self.include_memory else 'Disabled'}\n\n")
            
            # Enhanced table with memory
            if self.include_memory:
                f.write("| Model | Type | Latency (ms) | RTF | Memory (MB) | Notes |\n")
                f.write("|-------|------|--------------|-----|-------------|-------|\n")
            else:
                f.write("| Model | Type | Latency (ms) | RTF | Notes |\n")
                f.write("|-------|------|--------------|-----|-------|\n")
            
            for res in self.results:
                if "FAILED" in res.notes:
                    if self.include_memory:
                        f.write(f"| {res.model_name} | {res.task_type} | N/A | N/A | N/A | ❌ {res.notes} |\n")
                    else:
                        f.write(f"| {res.model_name} | {res.task_type} | N/A | N/A | ❌ {res.notes} |\n")
                else:
                    if self.include_memory:
                        mem_str = f"{res.memory_mb:.1f}" if isinstance(res.memory_mb, (int, float)) else str(res.memory_mb)
                        f.write(f"| {res.model_name} | {res.task_type} | {res.latency_ms:.0f} | {res.rtf:.4f} | {mem_str} | {res.notes} |\n")
                    else:
                        f.write(f"| {res.model_name} | {res.task_type} | {res.latency_ms:.0f} | {res.rtf:.4f} | {res.notes} |\n")
            
            f.write("\n\n**Metrics Explained:**\n")
            f.write("- **RTF (Real-Time Factor)**: < 1.0 means faster than real-time\n")
            if self.include_memory:
                f.write("- **Memory (MB)**: Peak memory usage during synthesis\n")
            if self.include_streaming:
                f.write("- **TTFA**: Time-To-First-Audio for streaming models\n")
        
        logger.info("Report saved successfully.")
        print(f"\nBenchmark complete! Results saved to {REPORT_FILE}")
        print(open(REPORT_FILE).read())



def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for voice cloning models")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], help="Force device usage")
    parser.add_argument("--skip-asr", action="store_true", help="Skip ASR benchmarks")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS benchmarks")
    parser.add_argument("--include-streaming", action="store_true", help="Include streaming benchmarks")
    parser.add_argument("--include-cloning", action="store_true", help="Include voice cloning benchmarks")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory tracking")
    parser.add_argument("--models", type=str, help="Comma-separated list of specific models to test")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        device=args.device,
        include_streaming=args.include_streaming,
        include_cloning=args.include_cloning,
        include_memory=not args.no_memory
    )
    
    # Generate test audio first (uses Kokoro)
    runner.generate_test_audio()

    # Helper to check if model should be run
    def should_run(name):
        if not args.models: return True
        return any(m.lower() in name.lower() for m in args.models.split(","))

    # TTS Benchmarks
    if not args.skip_tts:
        # Kitten
        if should_run("kitten"):
            try:
                from src.voice_cloning.tts.kitten_nano import KittenNanoTTS
                runner.benchmark_tts("KittenTTS (Nano)", KittenNanoTTS)
            except ImportError: logger.warning("Skipping KittenTTS: Not installed")
            except Exception as e: logger.warning(f"Skipping KittenTTS: {e}")

        # Kokoro
        if should_run("kokoro"):
            try:
                from src.voice_cloning.tts.kokoro import synthesize_speech
                runner.benchmark_tts("Kokoro", None)
            except ImportError: logger.warning("Skipping Kokoro: Not installed")
            except Exception as e: logger.warning(f"Skipping Kokoro: {e}")

        # Marvis
        if should_run("marvis"):
            try:
                from src.voice_cloning.tts.marvis import MarvisTTS
                runner.benchmark_tts("Marvis (MLX)", MarvisTTS)
            except ImportError: logger.warning("Skipping Marvis: Not installed")
            except Exception as e: logger.warning(f"Skipping Marvis: {e}")

        # Supertone
        if should_run("supertone"):
            try:
                from src.voice_cloning.tts.supertone import SupertoneTTS
                runner.benchmark_tts("Supertone", SupertoneTTS)
            except ImportError: logger.warning("Skipping Supertone: Not installed")
            except Exception as e: logger.warning(f"Skipping Supertone: {e}")
        
        # NeuTTS Air
        if should_run("neutts"):
            try:
                from src.voice_cloning.tts.neutts_air import NeuTTSAirTTS
                runner.benchmark_tts("NeuTTS Air", NeuTTSAirTTS)
            except ImportError: logger.warning("Skipping NeuTTS Air: Not installed")
            except Exception as e: logger.warning(f"Skipping NeuTTS Air: {e}")

    # ASR Benchmarks
    if not args.skip_asr:
        # Whisper
        if should_run("whisper"):
            try:
                from src.voice_cloning.asr.whisper import WhisperASR
                runner.benchmark_asr("Whisper (Large-v3)", WhisperASR, model_id="openai/whisper-large-v3")
            except ImportError: logger.warning("Skipping Whisper: Not installed")
            except Exception as e: logger.warning(f"Skipping Whisper: {e}")

        # Parakeet
        if should_run("parakeet"):
            try:
                from src.voice_cloning.asr.parakeet import ParakeetASR
                runner.benchmark_asr("Parakeet", ParakeetASR)
            except ImportError: logger.warning("Skipping Parakeet: Not installed")
            except Exception as e: logger.warning(f"Skipping Parakeet: {e}")

        # Canary
        if should_run("canary"):
            try:
                from src.voice_cloning.asr.canary import CanaryASR
                runner.benchmark_asr("Canary", CanaryASR)
            except ImportError: logger.warning("Skipping Canary: Not installed")
            except Exception as e: logger.warning(f"Skipping Canary: {e}")

    # VAD Benchmarks
    if should_run("humaware") or should_run("vad"):
        try:
            from src.voice_cloning.vad.humaware import HumAwareVAD
            runner.benchmark_vad("HumAware VAD", HumAwareVAD)
        except ImportError: logger.warning("Skipping HumAware: Not installed")
        except Exception as e: logger.warning(f"Skipping HumAware: {e}")

    runner.save_report()

if __name__ == "__main__":
    main()
