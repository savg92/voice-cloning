#!/usr/bin/env python3
"""
Multilingual Benchmark for TTS and ASR Models

Tests multilingual models with Spanish language to evaluate:
- TTS synthesis quality and performance
- ASR transcription accuracy and performance
- Cross-model comparison

Models tested:
- TTS: Chatterbox (multilingual), Kokoro (Spanish)
- ASR: Whisper, Canary, Parakeet
"""

import time
import logging
from pathlib import Path
from typing import Dict, List
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultilingualBenchmark:
    def __init__(self, output_dir: str = "outputs/multilingual_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Spanish test sentences
        self.spanish_tests = [
            "Hola, ¿cómo estás? El día está soleado y hermoso.",
            "La inteligencia artificial está cambiando el mundo de la tecnología.",
            "Me gusta mucho la música y el arte contemporáneo."
        ]
        
        self.results = []
    
    def benchmark_tts_spanish(self):
        """Benchmark Spanish TTS models"""
        logger.info("=" * 60)
        logger.info("BENCHMARKING SPANISH TTS MODELS")
        logger.info("=" * 60)
        
        test_text = self.spanish_tests[0]
        
        # 1. Chatterbox Multilingual
        logger.info("\n1. Testing Chatterbox (Multilingual - Spanish)")
        try:
            from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            
            output_path = self.output_dir / "chatterbox_spanish.wav"
            start_time = time.time()
            
            synthesize_with_chatterbox(
                text=test_text,
                output_wav=str(output_path),
                language="es",
                multilingual=True,
                exaggeration=0.7,
                cfg_weight=0.5
            )
            
            latency = time.time() - start_time
            info = sf.info(str(output_path))
            rtf = latency / info.duration
            
            self.results.append({
                "model": "Chatterbox (Spanish)",
                "type": "TTS",
                "language": "Spanish",
                "latency_s": latency,
                "rtf": rtf,
                "audio_duration_s": info.duration,
                "success": True
            })
            
            logger.info(f"✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, Duration={info.duration:.2f}s")
            
        except Exception as e:
            logger.error(f"✗ Chatterbox failed: {e}")
            self.results.append({
                "model": "Chatterbox (Spanish)",
                "type": "TTS",
                "language": "Spanish",
                "success": False,
                "error": str(e)
            })
        # 1b. Chatterbox (MLX - Spanish)
        logger.info("\n1b. Testing Chatterbox (MLX - Spanish)")
        try:
            from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            import os
            
            output_path = self.output_dir / "chatterbox_mlx_spanish.wav"
            start_time = time.time()
            
            output_path = self.output_dir / "chatterbox_mlx_spanish.wav"
            start_time = time.time()
            
            # Use provided reference or None
            ref_audio = "samples/anger.wav" if os.path.exists("samples/anger.wav") else None
            
            if not ref_audio:
                logger.info("  No reference audio found (samples/anger.wav). Testing zero-shot/default voice.")

            synthesize_with_chatterbox(
                text=test_text,
                output_wav=str(output_path),
                source_wav=ref_audio, 
                use_mlx=True,
                language="es"
            )
            
            latency = time.time() - start_time
            info = sf.info(str(output_path))
            rtf = latency / info.duration
            
            self.results.append({
                "model": "Chatterbox (MLX - Spanish)",
                "type": "TTS",
                "language": "Spanish",
                "latency_s": latency,
                "rtf": rtf,
                "audio_duration_s": info.duration,
                "success": True
            })
            
            logger.info(f"✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, Duration={info.duration:.2f}s")
            
        except Exception as e:
            logger.warning(f"  Chatterbox MLX skipped/failed: {e}")
            # Don't fail the whole benchmark if just MLX fails (might be CPU)
            if "MLX" in str(e) or "mlx" in str(e):
                 pass
            else:
                 self.results.append({
                    "model": "Chatterbox (MLX - Spanish)",
                    "type": "TTS",
                    "language": "Spanish",
                    "success": False,
                    "error": str(e)
                })
        logger.info("\n2. Testing Kokoro (Spanish)")
        try:
            from src.voice_cloning.tts.kokoro import synthesize_speech
            
            output_path = self.output_dir / "kokoro_spanish.wav"
            start_time = time.time()
            
            synthesize_speech(
                text=test_text,
                output_path=str(output_path),
                lang_code="e",  # 'e' for Spanish in Kokoro
                voice="af_heart",
                speed=1.0
            )
            
            latency = time.time() - start_time
            info = sf.info(str(output_path))
            rtf = latency / info.duration
            
            self.results.append({
                "model": "Kokoro (Spanish)",
                "type": "TTS",
                "language": "Spanish",
                "latency_s": latency,
                "rtf": rtf,
                "audio_duration_s": info.duration,
                "success": True
            })
            
            logger.info(f"✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, Duration={info.duration:.2f}s")
            
        except Exception as e:
            logger.error(f"✗ Kokoro failed: {e}")
            self.results.append({
                "model": "Kokoro (Spanish)",
                "type": "TTS",
                "language": "Spanish",
                "success": False,
                "error": str(e)
            })
    
        logger.info("\n3. Testing CosyVoice2 (MLX - Spanish)")
        try:
            from src.voice_cloning.tts.cosyvoice import synthesize_speech
            
            output_path = self.output_dir / "cosyvoice_mlx_spanish.wav"
            start_time = time.time()
            
            # Use reference audio if available (for zero-shot prompt)
            ref_audio = "samples/anger.wav" if Path("samples/anger.wav").exists() else None
            
            synthesize_speech(
                text=test_text,
                output_path=str(output_path),
                ref_audio_path=str(ref_audio) if ref_audio else None,
                use_mlx=True
            )
            
            latency = time.time() - start_time
            info = sf.info(str(output_path))
            rtf = latency / info.duration
            
            self.results.append({
                "model": "CosyVoice2 (MLX)",
                "type": "TTS",
                "language": "Spanish",
                "latency_s": latency,
                "rtf": rtf,
                "audio_duration_s": info.duration,
                "success": True
            })
            
            logger.info(f"✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, Duration={info.duration:.2f}s")
            
        except Exception as e:
            logger.warning(f"  CosyVoice2 MLX skipped/failed: {e}")
            if "MLX" not in str(e): # Record failure unless it's just missing MLX on non-Mac
                self.results.append({
                    "model": "CosyVoice2 (MLX)",
                    "type": "TTS",
                    "language": "Spanish",
                    "success": False,
                    "error": str(e)
                })

        logger.info("\n4. Testing CosyVoice2 (PyTorch - Spanish)")
        try:
            from src.voice_cloning.tts.cosyvoice import synthesize_speech
            
            output_path = self.output_dir / "cosyvoice_torch_spanish.wav"
            start_time = time.time()
            
            ref_audio = "samples/anger.wav" if Path("samples/anger.wav").exists() else None

            synthesize_speech(
                text=test_text,
                output_path=str(output_path),
                ref_audio_path=str(ref_audio) if ref_audio else None,
                use_mlx=False
            )
            
            latency = time.time() - start_time
            info = sf.info(str(output_path))
            rtf = latency / info.duration
            
            self.results.append({
                "model": "CosyVoice2 (PyTorch)",
                "type": "TTS",
                "language": "Spanish",
                "latency_s": latency,
                "rtf": rtf,
                "audio_duration_s": info.duration,
                "success": True
            })
            
            logger.info(f"✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, Duration={info.duration:.2f}s")
            
        except Exception as e:
            logger.warning(f"  CosyVoice2 PyTorch skipped/failed: {e}")
            self.results.append({
                "model": "CosyVoice2 (PyTorch)",
                "type": "TTS",
                "language": "Spanish",
                "success": False,
                "error": str(e)
            })

    def benchmark_asr_spanish(self):
        """Benchmark Spanish ASR models"""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARKING SPANISH ASR MODELS")
        logger.info("=" * 60)
        
        # Find Spanish audio files from TTS benchmark
        test_files = [
            (self.output_dir / "chatterbox_spanish.wav", self.spanish_tests[0]),
            (self.output_dir / "chatterbox_mlx_spanish.wav", self.spanish_tests[0]),
            (self.output_dir / "kokoro_spanish.wav", self.spanish_tests[0]),
            (self.output_dir / "cosyvoice_mlx_spanish.wav", self.spanish_tests[0]),
            (self.output_dir / "cosyvoice_torch_spanish.wav", self.spanish_tests[0])
        ]
        
        for audio_path, reference_text in test_files:
            if not audio_path.exists():
                logger.warning(f"Skipping {audio_path.name} - file not found")
                continue
            
            logger.info(f"\nTesting with: {audio_path.name}")
            logger.info(f"Reference: {reference_text}")
            
            # 1. Whisper Variants (Fast models only - skip slow standard Medium/Large-v3)
            logger.info("\n  Testing Whisper Variants...")
            
            whisper_variants = [
                {"name": "Whisper (Large-v3 Turbo)", "id": "openai/whisper-large-v3-turbo", "mlx": False},
                {"name": "Whisper (MLX Turbo)", "id": "mlx-community/whisper-large-v3-turbo", "mlx": True},
                {"name": "Whisper (MLX Medium)", "id": "mlx-community/whisper-medium", "mlx": True}
            ]
            
            from src.voice_cloning.asr.whisper import WhisperASR
            
            for variant in whisper_variants:
                logger.info(f"\n    Testing {variant['name']}...")
                try:
                    # Skip MLX if not on Mac
                    if variant['mlx']:
                        import platform
                        if platform.system() != "Darwin" or platform.machine() != "arm64":
                            logger.info(f"      Skipping {variant['name']} (MLX requires Apple Silicon)")
                            continue
                            
                    model = WhisperASR(device="mps", model_id=variant['id'], use_mlx=variant['mlx'])
                    if not variant['mlx']:
                        model.load_model()
                    
                    start_time = time.time()
                    result = model.transcribe(str(audio_path))
                    latency = time.time() - start_time
                    
                    transcription = result if isinstance(result, str) else result.get("text", "")
                    
                    info = sf.info(str(audio_path))
                    rtf = latency / info.duration
                    
                    # Calculate character error rate (simple)
                    cer = self._calculate_cer(reference_text, transcription)
                    
                    self.results.append({
                        "model": variant['name'],
                        "type": "ASR",
                        "language": "Spanish",
                        "audio_source": audio_path.name,
                        "latency_s": latency,
                        "rtf": rtf,
                        "audio_duration_s": info.duration,
                        "reference": reference_text,
                        "transcription": transcription,
                        "cer": cer,
                        "success": True
                    })
                    
                    logger.info(f"      ✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, CER={cer:.2%}")
                    logger.info(f"      Transcription: {transcription}")
                    
                except Exception as e:
                    logger.error(f"      ✗ {variant['name']} failed: {e}")
                    self.results.append({
                        "model": variant['name'],
                        "type": "ASR",
                        "language": "Spanish",
                        "audio_source": audio_path.name,
                        "success": False,
                        "error": str(e)
                    })
            
            # 2. Canary
            logger.info("\n  Testing Canary...")
            try:
                from src.voice_cloning.asr.canary import CanaryASR
                
                model = CanaryASR()
                
                start_time = time.time()
                result = model.transcribe(str(audio_path), source_lang="es", target_lang="es")
                latency = time.time() - start_time
                
                # Extract text from result dict
                transcription = result['text'] if isinstance(result, dict) else str(result)
                
                info = sf.info(str(audio_path))
                rtf = latency / info.duration
                
                cer = self._calculate_cer(reference_text, transcription)
                
                self.results.append({
                    "model": "Canary",
                    "type": "ASR",
                    "language": "Spanish",
                    "audio_source": audio_path.name,
                    "latency_s": latency,
                    "rtf": rtf,
                    "audio_duration_s": info.duration,
                    "reference": reference_text,
                    "transcription": transcription,
                    "cer": cer,
                    "success": True
                })
                
                logger.info(f"  ✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, CER={cer:.2%}")
                logger.info(f"  Transcription: {transcription}")
                
            except Exception as e:
                logger.error(f"  ✗ Canary failed: {e}")
                self.results.append({
                    "model": "Canary",
                    "type": "ASR",
                    "language": "Spanish",
                    "audio_source": audio_path.name,
                    "success": False,
                    "error": str(e)
                })
            
            # 3. Parakeet (Multilingual)
            logger.info("\n  Testing Parakeet...")
            try:
                from src.voice_cloning.asr.parakeet import ParakeetASR
                
                model = ParakeetASR(device="mps")
                
                start_time = time.time()
                transcription = model.transcribe(str(audio_path))
                latency = time.time() - start_time
                
                info = sf.info(str(audio_path))
                rtf = latency / info.duration
                
                cer = self._calculate_cer(reference_text, transcription)
                
                self.results.append({
                    "model": "Parakeet",
                    "type": "ASR",
                    "language": "Spanish",
                    "audio_source": audio_path.name,
                    "latency_s": latency,
                    "rtf": rtf,
                    "audio_duration_s": info.duration,
                    "reference": reference_text,
                    "transcription": transcription,
                    "cer": cer,
                    "success": True
                })
                
                logger.info(f"  ✓ Success: Latency={latency:.2f}s, RTF={rtf:.4f}, CER={cer:.2%}")
                logger.info(f"  Transcription: {transcription}")
                
            except Exception as e:
                logger.error(f"  ✗ Parakeet failed: {e}")
                self.results.append({
                    "model": "Parakeet",
                    "type": "ASR",
                    "language": "Spanish",
                    "audio_source": audio_path.name,
                    "success": False,
                    "error": str(e)
                })
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        if not reference or not hypothesis:
            return 1.0
        
        # Remove case and extra spaces
        ref = reference.lower().strip()
        hyp = hypothesis.lower().strip()
        
        # Simple character-level comparison
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        
        # Levenshtein distance (simplified)
        m, n = len(ref_chars), len(hyp_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n] / max(m, 1)
    
    def generate_report(self):
        """Generate markdown report"""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING REPORT")
        logger.info("=" * 60)
        
        report_path = self.output_dir / "MULTILINGUAL_RESULTS.md"
        
        with open(report_path, 'w') as f:
            f.write("# Multilingual Benchmark Results (Spanish)\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Test Language**: Spanish\n\n")
            
            # TTS Results
            f.write("## Text-to-Speech (TTS) Results\n\n")
            f.write("| Model | Success | Latency (s) | RTF | Audio Duration (s) |\n")
            f.write("|-------|---------|-------------|-----|-------------------|\n")
            
            for result in self.results:
                if result["type"] == "TTS":
                    if result["success"]:
                        f.write(f"| {result['model']} | ✅ | {result['latency_s']:.2f} | {result['rtf']:.4f} | {result['audio_duration_s']:.2f} |\n")
                    else:
                        f.write(f"| {result['model']} | ❌ | - | - | - |\n")
            
            # ASR Results
            f.write("\n## Automatic Speech Recognition (ASR) Results\n\n")
            f.write("| Model | Audio Source | Success | Latency (s) | RTF | CER | Transcription |\n")
            f.write("|-------|--------------|---------|-------------|-----|-----|---------------|\n")
            
            for result in self.results:
                if result["type"] == "ASR":
                    if result["success"]:
                        trans = result['transcription'][:50] + "..." if len(result['transcription']) > 50 else result['transcription']
                        f.write(f"| {result['model']} | {result['audio_source']} | ✅ | {result['latency_s']:.2f} | {result['rtf']:.4f} | {result['cer']:.2%} | {trans} |\n")
                    else:
                        f.write(f"| {result['model']} | {result['audio_source']} | ❌ | - | - | - | - |\n")
            
            # Reference Text
            f.write("\n## Reference Text\n\n")
            f.write(f"```\n{self.spanish_tests[0]}\n```\n\n")
            
            # Analysis
            f.write("## Analysis\n\n")
            f.write("### TTS Models\n")
            tts_success = [r for r in self.results if r["type"] == "TTS" and r["success"]]
            if tts_success:
                fastest = min(tts_success, key=lambda x: x["latency_s"])
                f.write(f"- **Fastest**: {fastest['model']} ({fastest['latency_s']:.2f}s, {fastest['rtf']:.4f}x RTF)\n")
            
            f.write("\n### ASR Models\n")
            asr_success = [r for r in self.results if r["type"] == "ASR" and r["success"]]
            if asr_success:
                most_accurate = min(asr_success, key=lambda x: x["cer"])
                fastest_asr = min(asr_success, key=lambda x: x["latency_s"])
                f.write(f"- **Most Accurate**: {most_accurate['model']} ({most_accurate['cer']:.2%} CER)\n")
                f.write(f"- **Fastest**: {fastest_asr['model']} ({fastest_asr['latency_s']:.2f}s)\n")
        
        logger.info(f"Report saved to: {report_path}") # Keep original logging, as args.output is not defined here.
        return report_path
    
    def run(self):
        """Run all benchmarks"""
        logger.info("Starting Multilingual Benchmark (Spanish)")
        
        # Run TTS benchmarks
        self.benchmark_tts_spanish()
        
        # Run ASR benchmarks
        self.benchmark_asr_spanish()
        
        # Generate report
        report_path = self.generate_report()
        
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK COMPLETE")
        logger.info(f"Results: {report_path}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    benchmark = MultilingualBenchmark()
    benchmark.run()
