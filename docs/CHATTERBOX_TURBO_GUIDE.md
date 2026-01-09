# Chatterbox Turbo TTS - High-Speed Synthesis Guide

Chatterbox Turbo is a highly optimized variant of the expressive Chatterbox text-to-speech model. It is designed for maximum speed and lower latency while maintaining the expressive characteristics of the original model.

## 🚀 Key Features
- **High Speed**: Significantly faster than the standard Chatterbox model (~2-3x speedup).
- **Expressive**: Maintains support for emotion intensity via exaggeration.
- **Zero-Shot Cloning**: Supports voice cloning from a short reference audio clip.
- **Apple Silicon Optimized**: Native MLX support for blazing fast performance on Mac.
- **English Focused**: Primarily optimized for English speech (multilingual support falls back to English in the current Turbo version).

---

## 🍎 MLX Backend (Highly Recommended for Mac)

The MLX backend for Chatterbox Turbo provides the best performance on Apple Silicon (M1/M2/M3). It utilizes 4-bit quantization to reduce memory usage and increase throughput.

### Usage
Use the `--model chatterbox-turbo` and `--use-mlx` flags.

```bash
# Basic Synthesis (using default voice)
uv run python main.py --model chatterbox-turbo --text "This is a high-speed test." --use-mlx

# Voice Cloning with Turbo
uv run python main.py --model chatterbox-turbo --text "Cloning at high speed." --reference samples/anger.wav --use-mlx

# Custom Voice Preset & Speed
uv run python main.py --model chatterbox-turbo --text "Speaking quickly." --voice af_heart --speed 1.2 --use-mlx
```

---

## 🐍 PyTorch Backend

The PyTorch backend uses the `chatterbox-turbo` weights and an optimized `inference_turbo` logic.

### Usage
```bash
# Basic Synthesis
uv run python main.py --model chatterbox-turbo --text "Synthesizing with Turbo PyTorch."

# Expressive Control
uv run python main.py --model chatterbox-turbo --text "Wow, this is fast!" --exaggeration 0.9
```

---

## 🛠️ Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--text` | string | required | Text to synthesize. |
| `--reference` | string | optional | Path to audio file for zero-shot cloning. |
| `--use-mlx` | flag | false | Use optimized MLX backend (recommended for Mac). |
| `--voice` | string | `af_heart` | Voice preset to use (when not cloning). |
| `--exaggeration`| float | `0.5` | Emotion intensity (0-1). |
| `--speed` | float | `1.0` | Playback speed multiplier (MLX only). |

---

## 📊 Turbo vs Standard Comparison

| Feature | Standard | Turbo |
|---------|----------|-------|
| **Latency** | Moderate (~7s) | Low (~2.7s) |
| **Throughput** | 1.0x | ~2.5x |
| **Languages** | 23 | English (primarily) |
| **Best For** | Quality, Multilingual | Real-time, Performance |

---

## 💡 Best Practices
- **Use MLX on Mac**: The speed difference on Apple Silicon is massive.
- **English Only**: If you need high-quality Spanish, French, or other languages, stick to the **Standard** `chatterbox` model.
- **Cloning**: For cloning, provide a clean reference clip of 5-10 seconds. Turbo is great for interactive applications where cloning latency matters.
