# Benchmark Guide

Comprehensive guide for the Voice Cloning Benchmark Suite, which evaluates performance of TTS, ASR, and VAD models.

## Overview

The benchmark suite measures:
- **Latency**: Time taken to process input
- **RTF (Real-Time Factor)**: Ratio of processing time to audio duration (< 1.0 is faster than real-time)
- **Memory Usage**: Peak memory consumption during processing
- **TTFA (Time-To-First-Audio)**: Latency until first audio chunk (streaming models)

## Quick Start

### Basic Benchmark

Run all available benchmarks with default settings:

```bash
python benchmark.py
```

### Selective Benchmarks

Skip specific types:

```bash
# Skip ASR benchmarks
python benchmark.py --skip-asr

# Skip TTS benchmarks  
python benchmark.py --skip-tts

# Only run TTS benchmarks
python benchmark.py --skip-asr
```

### Force Specific Device

```bash
# Force CPU
python benchmark.py --device cpu

# Force MPS (Apple Silicon)
python benchmark.py --device mps

# Force CUDA (NVIDIA GPU)
python benchmark.py --device cuda
```

## Advanced Features

### Streaming Benchmarks

Enable streaming tests to measure Time-To-First-Audio (TTFA):

```bash
python benchmark.py --include-streaming
```

Streaming tests measure:
- Time until first audio chunk is generated
- Average chunk generation time
- Total latency vs. TTFA comparison

**Supported Models:**
- Supertone (pseudo-streaming via sentence splitting)
- KittenTTS (pseudo-streaming via sentence splitting)
- Kokoro (native chunk-based streaming)
- Marvis (has streaming API)

### Voice Cloning Benchmarks

Test voice cloning capabilities with reference audio:

```bash
python benchmark.py --include-cloning
```

Voice cloning tests:
- Use generated reference audio for cloning
- Measure cloning latency and quality
- Compare with non-cloning synthesis

**Supported Models:**
- Marvis TTS (voice cloning via reference audio)
- NeuTTS Air (voice cloning TTS)

### Memory Tracking

Memory tracking is enabled by default. To disable:

```bash
python benchmark.py --no-memory
```

Memory tracking shows:
- Peak memory usage during synthesis/transcription
- Memory overhead per model
- Helps identify memory-intensive models

### Test Specific Models

Use `--models` to test only specific models:

```bash
# Test only Kitten and Kokoro
python benchmark.py --models kitten,kokoro

# Test only streaming-capable models
python benchmark.py --models supertone,kokoro,marvis --include-streaming
```

## Metrics Explained

### RTF (Real-Time Factor)

RTF = Processing Time / Audio Duration

- **RTF < 1.0**: Faster than real-time (✅ Good for real-time apps)
- **RTF = 1.0**: Exactly real-time
- **RTF > 1.0**: Slower than real-time (⚠️ Not suitable for real-time)

**Example:**
- Audio duration: 10 seconds
- Processing time: 2 seconds
- RTF = 2 / 10 = 0.2 (5× faster than real-time)

### Latency

Total time from input to output completion.

- **Low latency** (< 1s): Excellent for interactive applications
- **Medium latency** (1-3s): Acceptable for most use cases
- **High latency** (> 3s): Better for batch processing

### Memory Usage

Peak memory consumed during processing.

- Helps identify resource requirements
- Important for deployment planning
- Useful for comparing models

### TTFA (Time-To-First-Audio)

Time until the first audio chunk is available (streaming only).

- **Low TTFA** (< 500ms): Feels instantaneous
- **Medium TTFA** (500ms-1s): Acceptable
- **High TTFA** (> 1s): Noticeable delay

## Understanding Results

### Sample Output

```markdown
| Model | Type | Latency (ms) | RTF | Memory (MB) | Notes |
|-------|------|--------------|-----|-------------|-------|
| Supertone | TTS | 583 | 0.0838 | 245.3 | Device: mps |
| KittenTTS (Nano) | TTS | 1255 | 0.1625 | 180.7 | Device: mps |
| Kokoro | TTS | 3140 | 0.4423 | 312.5 | Device: mps |
```

**Analysis:**
- **Supertone**: Fastest (12× real-time), low memory
- **KittenTTS**: Very fast (6× real-time), lowest memory
- **Kokoro**: Fast (2× real-time), moderate memory

## Benchmarked Models

### TTS Models

| Model | Type | Streaming | Cloning | Notes |
|-------|------|-----------|---------|-------|
| **KittenTTS Nano** | Fast TTS | ✅ Pseudo | ❌ | CPU-friendly, English only |
| **Kokoro** | Neural TTS | ✅ Native | ❌ | Multilingual, high quality |
| **Marvis** | MLX TTS | ✅ Native | ✅ | Apple Silicon optimized |
| **Supertone** | ONNX TTS | ✅ Pseudo | ❌ | Ultra-fast, lightweight |
| **NeuTTS Air** | Voice Cloning | ❌ | ✅ | Cloning only (requires ref) |

### ASR Models

| Model | Type | Device | Notes |
|-------|------|--------|-------|
| **Whisper Large-v3** | Transformer ASR | GPU/CPU | Most accurate |
| **Parakeet** | RNN-T ASR | MPS/GPU | Fast, streaming-capable |
| **Canary** | Multilingual ASR | GPU/CPU | 100+ languages |

### VAD Models

| Model | Type | Notes |
|-------|------|-------|
| **HumAware VAD** | Silero-based | Speech vs. humming detection |

## Troubleshooting

### HumAware VAD Fails

**Error**: `Could not load libtorchcodec`

**Solution**: The `torchcodec` dependency has been removed. Update your environment:

```bash
uv pip install -e .
```

### Model Not Found

**Error**: `ModuleNotFoundError` or `Skipping X: Not installed`

**Solution**: Install the missing model:

```bash
# For specific models
uv pip install kokoro kittentts

# For all dependencies
uv pip install -e .
```

### Out of Memory

**Error**: System runs out of memory

**Solutions:**
1. Use `--device cpu` to reduce memory usage
2. Test models individually: `--models kokoro`
3. Skip memory tracking: `--no-memory`
4. Close other applications

### Slow Performance

**Issue**: Benchmarks taking too long

**Solutions:**
1. Skip ASR benchmarks: `--skip-asr` (ASR models are slower)
2. Test specific models: `--models supertone,kitten`
3. Use GPU/MPS if available: `--device mps` or `--device cuda`

## Best Practices

1. **Warm Start**: Run benchmarks twice - first run downloads models and compiles
2. **Consistent Environment**: Close other applications for accurate memory tracking
3. **Device Selection**: Use `--device` to ensure consistent comparisons
4. **Batch Testing**: Use `--models` to test groups of similar models
5. **Document Results**: Save benchmark reports with timestamps for comparison

## Output Files

- **`BENCHMARK_RESULTS.md`**: Main results table with all metrics
- **`outputs/benchmark/`**: Generated audio files for verification
  - `benchmark_reference.wav`: Test audio for ASR/VAD
  - `bench_*.wav`: Generated TTS outputs
  - `warmup_*.wav`: Warmup outputs (can be deleted)

## Example Workflows

### Compare TTS Models

```bash
# Test all TTS models with memory tracking
python benchmark.py --skip-asr

# Include streaming metrics
python benchmark.py --skip-asr --include-streaming
```

### Test Streaming Performance

```bash
# Only streaming-capable models
python benchmark.py --models supertone,kokoro,kitten,marvis --include-streaming
```

### Full Comprehensive Benchmark

```bash
# All tests (takes longest time)
python benchmark.py --include-streaming --include-cloning
```

### Quick Performance Check

```bash
# Fast models only, no ASR
python benchmark.py --models supertone,kitten --skip-asr --no-memory
```

## Interpreting for Production

### For Real-Time Applications
- **Prioritize**: RTF < 0.5, TTFA < 500ms
- **Recommended**: Supertone, KittenTTS

### For High Quality
- **Prioritize**: Audio quality over speed
- **Recommended**: Kokoro, Marvis

### For Memory-Constrained Devices
- **Prioritize**: Low memory usage
- **Recommended**: KittenTTS, Supertone

### For Voice Cloning
- **Required**: Cloning support
- **Recommended**: Marvis, NeuTTS Air

## Contributing

To add a new model to benchmarks:

1. Create wrapper class in `src/voice_cloning/tts/` or similar
2. Add benchmark call in `benchmark.py` main()
3. Handle model-specific initialization in `benchmark_tts()`
4. Update this guide with model details

## See Also

- **[README.md](../README.md)**: Project overview and setup
- **[SUPERTONE_GUIDE.md](SUPERTONE_GUIDE.md)**: Supertone TTS guide
- **[KITTEN_GUIDE.md](KITTEN_GUIDE.md)**: KittenTTS guide
- **[KOKORO_GUIDE.md](KOKORO_GUIDE.md)**: Kokoro TTS guide
- **[MARVIS_GUIDE.md](MARVIS_GUIDE.md)**: Marvis TTS guide
