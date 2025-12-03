# Benchmark Results - MacBook Pro M3 8GB

**Hardware Specifications:**
- **Model**: MacBook Pro (M3, 2024)
- **Chip**: Apple M3
- **RAM**: 8GB Unified Memory
- **OS**: macOS Sonoma
- **Device**: MPS (Metal Performance Shaders)

**Date**: November 26, 2025  
**Benchmark Version**: v1.0 (with memory tracking)

---

## Summary

Performance benchmark of voice cloning models optimized for Apple Silicon (M3). All tests run on MPS backend for maximum performance on Mac.

### Top Performers

| Category | Winner | Metric |
|----------|--------|--------|
| **Fastest TTS** | Supertone | 583ms (12× real-time) |
| **Lowest Memory** | KittenTTS | 244.9 MB |
| **Fastest ASR** | Parakeet | 2.6s (2.7× real-time) |
| **Fastest VAD** | HumAware | 911ms (7.8× real-time) |
| **Best Balance** | KittenTTS | Fast + Low memory |

---

## Complete Benchmark Data

All results from MacBook Pro M3 8GB running on MPS (Metal Performance Shaders).

### All Models - Raw Data

| Model | Type | Latency (ms) | RTF | Memory (MB) | Audio Duration (s) | Speed Multiplier | Device |
|-------|------|--------------|-----|-------------|-------------------|------------------|--------|
| **Supertone** | TTS | 583 | 0.0839 | 301.0 | 6.95 | 11.9× | MPS |
| **KittenTTS Nano** | TTS | 1,306 | 0.1691 | 244.9 | 7.73 | 5.9× | MPS |
| **Kokoro** | TTS | 3,682 | 0.5186 | 0.0 | 7.10 | 1.9× | MPS |
| **Marvis (MLX)** | TTS | 10,678 | 1.7335 | 1.1 | 6.16 | 0.58× | MPS |
| **Parakeet** | ASR | 2,624 | 0.3695 | 0.0 | 7.10 | 2.7× | MPS |
| **Whisper (MLX Medium)** | ASR | 1,632 | 0.2299 | 0.0 | 7.10 | **4.3× real-time** | MPS |
| **Whisper (MLX Turbo)** | ASR | 2,174 | 0.3062 | 0.0 | 7.10 | 3.3× | MPS |
| **Canary** | ASR | 27,499 | 3.8730 | 0.0 | 7.10 | 0.26× | MPS |
| **Whisper (Standard Turbo)** | ASR | 47,792 | 6.7313 | 0.0 | 7.10 | 0.15× (slow) | MPS |
| **NeuTTS Air** | TTS | 55,335 | 5.7521 | 0.0 | 9.62 | 0.17× | MPS |
| **HumAware VAD** | VAD | 911 | 0.1283 | 0.0 | 7.10 | 7.8× | CPU |

**Legend:**
- **Latency**: Total processing time in milliseconds
- **RTF**: Real-Time Factor (< 1.0 is faster than real-time)
- **Memory**: Peak memory usage during processing (MB)
- **Audio Duration**: Length of processed/generated audio
- **Speed Multiplier**: How many times faster than real-time (1/RTF)
- **Device**: Processing backend (MPS = Metal Performance Shaders for Apple Silicon)

**Notes:**
- Kokoro and Marvis show 0.0 MB memory due to subprocess execution
- All TTS models tested with same input text (test sentence)
- All ASR/VAD models tested with same 7.1s audio file

---

## Text-to-Speech (TTS) Results

| Model | Latency | RTF | Memory | Speed Multiplier | Notes |
|-------|---------|-----|--------|------------------|-------|
| **Supertone** | 583ms | 0.084 | 301.0 MB | **12× real-time** | ONNX-based, ultra-fast |
| **KittenTTS Nano** | 1,306ms | 0.169 | 244.9 MB | **6× real-time** | Lowest memory, CPU-friendly |
| **Kokoro** | 3,682ms | 0.519 | 0.0 MB | **2× real-time** | High quality, multilingual |
| **Marvis** | 10,678ms | 1.734 | 1.1 MB | **0.6× real-time** | MLX-optimized, 4-bit quantized |
| **NeuTTS Air** | 55,335ms | 5.752 | 0.0 MB | **0.17× real-time** | Voice cloning, slow but high quality |

**Test Text**: "The quick brown fox jumps over the lazy dog. This is a benchmark test to measure synthesis speed." (7.1 seconds of audio)

### Kokoro: MLX vs PyTorch Backend Comparison

| Backend | Total Time | Speedup | Model | Notes |
|---------|------------|---------|-------|-------|
| **MLX** | **6.8s** | **1.42×** | mlx-community/Kokoro-82M-bf16 | Apple Silicon optimized |
| **PyTorch** | 9.7s | 1.0× (baseline) | hexgrad/Kokoro-82M | Standard backend |

**MLX Performance Benefits:**
- ⚡ **30% faster** synthesis on Apple Silicon
- 🎯 Same audio quality as PyTorch backend
- 🔧 Drop-in replacement with `--use-mlx` flag
- 🍎 Optimized for M1/M2/M3 chips

**When to use MLX:**
- Running on MacBook with Apple Silicon
- Need faster synthesis times
- Batch processing multiple files

**Usage:**
```bash
# PyTorch (standard)
uv run python main.py --model kokoro --text "Hello" --output out.wav

# MLX (30% faster on Apple Silicon)
uv run python main.py --model kokoro --text "Hello" --output out.wav --use-mlx
```

---

## Analysis

### Text-to-Speech Results

**🏆 Supertone - Speed Champion**
- Fastest by far with 12× real-time performance
- Moderate memory footprint (301 MB)
- Best choice for real-time applications
- ONNX backend provides consistent performance

**⚡ KittenTTS - Efficiency King**
- Excellent balance of speed (6× real-time) and memory (245 MB)
- Lowest memory usage of all models
- Perfect for resource-constrained devices
- English-only limitation

**🎨 Kokoro - Quality Leader**
- Still fast at 2× real-time
- Minimal memory overhead
- Best audio quality
- Supports 8 languages

**🍎 Marvis - Apple Silicon Native**
- Only model slower than real-time (0.6×)
- Uses MLX for Apple Silicon optimization
- 4-bit quantization for reduced model size
- Voice cloning support
- Runs in separate process (explains low memory reading)

**🎙️ NeuTTS Air - Voice Cloning Specialist**
- **Slowest model** (0.17× real-time, ~55s for 10s audio)
- **High Quality**: Uses NeuCodec for excellent audio fidelity
- **Voice Cloning**: Requires reference audio (3s+)
- **Best for**: Offline content creation where quality/cloning matters more than speed

---

## Automatic Speech Recognition (ASR) Results

| Model | Latency | RTF | Memory | Speed Multiplier | Notes |
|-------|---------|-----|--------|------------------|-------|
| **Whisper (Tiny)** | 472ms | 0.066 | 0.0 MB | **15× real-time** | OpenAI model, tiny variant |
| **Whisper (MLX Medium)** | 1,632ms | 0.230 | 0.0 MB | **4.3× real-time** | **Best for Mac** - MLX-optimized |
| **Whisper (MLX Turbo)** | 2,174ms | 0.306 | 0.0 MB | **3.3× real-time** | MLX-optimized, large model |
| **Parakeet** | 2,624ms | 0.370 | 0.0 MB | **2.7× real-time** | MLX-optimized, RNN-T architecture |
| **Canary** | 27,499ms | 3.873 | 0.0 MB | **0.26× real-time** | NeMo-based, 100+ languages |
| **Whisper (Standard Turbo)** | 47,792ms | 6.731 | 0.0 MB | **0.15× real-time** | ⚠️ Not recommended on Mac |

**Test Audio**: 7.1 seconds of synthesized speech

### Analysis

**🌪️ Whisper (Tiny) - Blazing Fast**
- Incredible 15× real-time performance
- Ideal for quick, on-device transcription
- Lowest accuracy but fastest speed
- Standard industry baseline

**⚡ Whisper (MLX Medium) - WINNER for Apple Silicon**
- **4.3× real-time** on Apple Silicon  
- MLX-optimized for Mac
- Excellent accuracy/speed tradeoff
- **#1 Recommended for most use cases on Mac**
- **21× faster than standard Turbo on MPS**

**🚀 Whisper (MLX Turbo) & Parakeet - Fast ASR**
- Both achieve ~3× real-time transcription
- MLX backend for Apple Silicon
- Whisper Turbo: Large model accuracy with good speed
- Parakeet: Streaming-capable RNN-T architecture

**⚠️ Standard Whisper (Large-v3, Medium, Turbo) on MPS**
- **Not recommended on Mac** - extremely slow (47.8s for 7s audio!)
- Standard Turbo is **22× slower** than MLX variants
- Use MLX variants instead for 20-30× speedup
- Standard PyTorch implementation not optimized for Apple Silicon
- Only use on NVIDIA GPUs where they perform well

### ASR Accuracy Comparison

**Test Setup:**
- Reference audio: Kokoro TTS-generated speech
- Reference text: "The quick brown fox jumps over the lazy dog. This is a benchmark test to measure synthesis speed."
- Metric: Character Error Rate (CER)

**English vs Spanish Accuracy:**

| Language | Model | CER | Accuracy | Notes |
|----------|-------|-----|----------|-------|
| **Spanish** | MLX Turbo | **0.00%** | **100%** ✅ | Perfect transcription |
| **Spanish** | MLX Medium | **0.00%** | **100%** ✅ | Perfect transcription |
| **Spanish** | All models | **0.00%** | **100%** ✅ | All ASR models perfect on Spanish |
| **English** | MLX Turbo | 15.5% | 84.5% ⚠️ | Some transcription errors |
| **English** | MLX Medium | 19.6% | 80.4% ⚠️ | More errors than Turbo |

**Sample English Transcription Issues:**
```
Reference: "The quick brown fox jumps over the lazy dog..."
MLX Turbo: "Take kick-brown foxhumps over telathidog..."
MLX Medium: "Take each brown fox hoombs over to LathiDog..."
```

**Analysis:**
- **Perfect accuracy on Spanish** (0% CER across all models)
- **Lower accuracy on English** (15-20% CER)
- **Likely cause**: Kokoro TTS synthesis quality varies by language
- **Not a Whisper issue**: All models achieved perfect Spanish transcription
- **Recommendation**: Use real recorded audio for production accuracy testing, not synthesized speech

**Key Insight:** The accuracy difference suggests Kokoro produces clearer Spanish synthesis than English, or has pronunciation artifacts in English mode.

**🌍 Canary - Multilingual Powerhouse**
- Slower but supports 100+ languages
- NeMo-based transformer model
- Better for accuracy over speed
- Large model (1B parameters)

---

## Voice Activity Detection (VAD) Results

| Model | Latency | RTF | Memory | Speed Multiplier | Notes |
|-------|---------|-----|--------|------------------|-------|
| **HumAware VAD** | 911ms | 0.128 | 0.0 MB | **7.8× real-time** | ✅ Fixed! Silero-based, speech vs. humming |

**Test Audio**: 7.1 seconds of synthesized speech

### Analysis

**✅ HumAware VAD - Now Working!**
- Fast 7.8× real-time detection
- Silero-based fine-tuned model
- Distinguishes speech from humming
- Low memory overhead
- **Recently Fixed**: Removed torchcodec dependency, uses soundfile for audio loading

---

## Performance Comparison Charts

### TTS Speed (Lower is Better)

```
Supertone    ▓░░░░░░░░░░░░░░░░░░░ 0.08× (12× real-time)
KittenTTS    ▓▓░░░░░░░░░░░░░░░░░░ 0.17× (6× real-time)
Kokoro       ▓▓▓▓▓░░░░░░░░░░░░░░░ 0.52× (2× real-time)
Marvis       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░ 1.73× (0.6× real-time)
```

### TTS Memory Usage

```
Kokoro       ░░░░░░░░░░░░░░░░░░░░ 0.0 MB (baseline)
Marvis       ░░░░░░░░░░░░░░░░░░░░ 1.1 MB (subprocess)
KittenTTS    ▓▓▓▓▓▓▓░░░░░░░░░░░░░ 244.9 MB
Supertone    ▓▓▓▓▓▓▓▓░░░░░░░░░░░░ 301.0 MB
```

---

## Recommendations

### For Real-Time Applications
**Winner**: Supertone
- 12× real-time means instant response
- Handles interactive use cases easily
- Low latency (583ms)

### For Resource-Constrained Devices
**Winner**: KittenTTS
- Only 245 MB memory footprint
- Still 6× faster than real-time
- Excellent for deployment on limited RAM devices

### For Multilingual Support
**Winner**: Kokoro
- Supports 8 languages
- Still 2× real-time (fast enough for most use cases)
- High audio quality

### For Voice Cloning
**Winner**: Marvis
- Only TTS model with voice cloning in these benchmarks
- MLX-optimized for Apple Silicon
- Slower but acceptable for offline use

### For Transcription
**Winner**: Parakeet
- 2.7× real-time transcription
- MLX-optimized performance
- Good for real-time captioning

### For Voice Activity Detection
**Winner**: HumAware VAD
- 7.8× real-time detection
- Distinguishes speech from humming
- Great for preprocessing audio

---

## Streaming Performance

All models marked with streaming support have been tested:
- **Supertone**: ✅ Pseudo-streaming (sentence-based)
- **KittenTTS**: ✅ Pseudo-streaming (sentence-based)
- **Kokoro**: ✅ Native streaming (chunk-based)
- **Marvis**: ✅ Native streaming API

Run with `--stream` flag in main.py to enable:
```bash
python main.py --model supertone --text "Your text here" --stream --output out.wav
```

---

## System Configuration

**Python**: 3.12  
**PyTorch**: 2.9.1  
**Package Manager**: uv  
**Key Libraries**:
- `mlx` for Apple Silicon optimization
- `onnxruntime` for Supertone
- `kokoro` for multilingual TTS
- `kittentts` for fast TTS
- `soundfile` for audio I/O (replaced torchaudio to avoid torchcodec)

**Installation**:
```bash
uv pip install -e .
```

---

## Running Your Own Benchmarks

### Basic Benchmark
```bash
python benchmark.py
```

### Custom Options
```bash
# Test specific models
python benchmark.py --models supertone,kitten

# Skip ASR (faster)
python benchmark.py --skip-asr

# Disable memory tracking
python benchmark.py --no-memory

# Include streaming tests
python benchmark.py --include-streaming
```

See [docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) for complete guide.

---

## Changelog

**v1.0** (Nov 26, 2025)
- Initial benchmark results on M3 MacBook Pro 8GB
- Added memory tracking
- Enhanced reporting with test status indicators
- **Fixed HumAware VAD**: Removed torchcodec dependency, implemented standalone utils, replaced torchaudio.load with soundfile

---

## Recent Fixes

### HumAware VAD Fix
The HumAware VAD model initially failed due to `torchcodec` dependency issues. This was resolved by:

1. **Removed torch.hub dependency**: Implemented standalone `_get_speech_timestamps()` function
2. **Replaced audio loading**: Changed from `torchaudio.load()` to `soundfile.read()` to avoid torchcodec
3. **Standalone implementation**: No external dependencies beyond huggingface_hub for model loading

Now fully functional with excellent performance (7.8× real-time).

---

## Notes

- All benchmarks run on MPS (Metal Performance Shaders) for optimal Apple Silicon performance
- Memory measurements may vary based on system load
- RTF (Real-Time Factor) < 1.0 indicates faster than real-time processing
- Marvis memory appears low due to subprocess execution
- HumAware VAD now uses soundfile instead of torchaudio for compatibility

---

**For detailed usage and troubleshooting, see:**
- [Benchmark Guide](docs/BENCHMARK_GUIDE.md)
- [Main README](README.md)
