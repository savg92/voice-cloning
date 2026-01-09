import time
import logging
import psutil
import torch
import gc
import soundfile as sf
import os
from pathlib import Path
from dataclasses import dataclass

from .config import OUTPUT_DIR, ensure_output_dir, BENCHMARK_FILE
from .base import ModelBenchmark, BenchmarkType

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    model: str
    type: str
    latency_ms: float
    rtf: float
    memory_mb: float
    notes: str = ""
    audio_duration: float = 0.0
    device: str = "cpu"

@dataclass
class BenchmarkConfig:
    include_streaming: bool = False
    include_cloning: bool = False
    include_memory: bool = True
    output_dir: Path = OUTPUT_DIR
    reference_audio_path: str | None = None # For ASR/VAD
    test_text: str = ""

class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []
        ensure_output_dir()

    def _get_memory_usage(self):
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return 0.0

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def save_report(self):
        if not self.results:
            logger.info("No results to save.")
            return

        from .results_manager import BenchmarkResultsManager
        manager = BenchmarkResultsManager(BENCHMARK_FILE)
        
        for r in self.results:
            manager.update_result(r, audio_duration=r.audio_duration, device=r.device)
            
        logger.info(f"Report updated in {BENCHMARK_FILE}")

    def run_benchmark(self, benchmark: ModelBenchmark):
        logger.info(f"Benchmarking: {benchmark.model_name}...")
        
        try:
            # Check prerequisites
            input_data = self.config.test_text
            if benchmark.type in [BenchmarkType.ASR, BenchmarkType.VAD]:
                if not self.config.reference_audio_path or not os.path.exists(self.config.reference_audio_path):
                    raise ValueError(f"{benchmark.type} benchmark requires reference audio path")
                input_data = self.config.reference_audio_path
            
            # Load
            start_load = time.time()
            benchmark.load()
            load_time = time.time() - start_load
            logger.info(f"  Load time: {load_time:.2f}s")
            
            # Warmup
            logger.info("  Warming up...")
            benchmark.warmup(str(self.config.output_dir))
            
            # Memory Baseline
            mem_before = self._get_memory_usage() if self.config.include_memory else 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            # Run Test
            logger.info(f"  Running latency test ({benchmark.type})...")
            start_time = time.time()
            
            output_filename = f"bench_{benchmark.model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
            if benchmark.type == BenchmarkType.TTS:
                output_filename += ".wav"
            else:
                output_filename += ".txt"
            
            output_path = str(self.config.output_dir / output_filename)
            
            result_data = benchmark.run_test(input_data, output_path)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            # Calculate RTF
            duration = 0
            if benchmark.type == BenchmarkType.TTS:
                 if 'audio' in result_data and 'sr' in result_data:
                     duration = len(result_data['audio']) / result_data['sr']
            elif benchmark.type in [BenchmarkType.ASR, BenchmarkType.VAD]:
                # For ASR/VAD, duration is input duration
                try:
                    info = sf.info(input_data)
                    duration = info.duration
                except Exception:
                    duration = 0 # Fallback
            
            rtf = (latency / 1000) / duration if duration > 0 else 0
            
            # Memory Peak
            mem_after = self._get_memory_usage() if self.config.include_memory else 0
            mem_peak = max(0, mem_after - mem_before)
            
            logger.info(f"  Latency: {latency:.2f}ms")
            logger.info(f"  RTF: {rtf:.4f}")
            logger.info(f"  Memory: {mem_peak:.2f}MB")
            
            notes = f"Device: {self._get_device()}"
            if 'error' in result_data:
                notes = f"WARNING: {result_data.get('error')}"
            
            self.add_result(BenchmarkResult(
                model=benchmark.model_name,
                type=benchmark.type,
                latency_ms=latency,
                rtf=rtf,
                memory_mb=mem_peak,
                notes=notes,
                audio_duration=duration,
                device=self._get_device().upper()
            ))
            
            benchmark.cleanup()
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            self.add_result(BenchmarkResult(
                model=benchmark.model_name,
                type=benchmark.type,
                latency_ms=0, rtf=0, memory_mb=0,
                notes=f"FAILED: {str(e)}",
                audio_duration=0,
                device=self._get_device().upper()
            ))