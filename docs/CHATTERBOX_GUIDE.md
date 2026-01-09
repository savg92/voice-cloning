# Chatterbox TTS - Installation & Usage

Chatterbox is an expressive text-to-speech model supporting 23 languages and zero-shot voice cloning. It is available in both standard PyTorch and optimized MLX backends.

## 🍎 MLX Backend (Recommended for Mac)

The MLX backend is highly optimized for Apple Silicon (M1/M2/M3) and is significantly faster than the standard PyTorch implementation on these devices.

### Prerequisites
Ensure you have `mlx-audio` installed:
```bash
uv pip install -U mlx-audio
```

### Usage
Use the `--use-mlx` flag to enable the optimized backend.

```bash
# Basic Synthesis
uv run python main.py --model chatterbox --text "Hello from MLX!" --use-mlx

# Voice Cloning
uv run python main.py --model chatterbox --text "Cloning a voice." --reference samples/anger.wav --use-mlx
```

---

## 🐍 PyTorch Backend (Standard)

The standard backend uses the `chatterbox-tts` library.

### Installation
```bash
uv pip install chatterbox-tts s3tokenizer diffusers pykakasi spacy-pkuseg conformer --no-deps
```

### Usage
```bash
# Basic Synthesis
uv run python main.py --model chatterbox --text "Hello from PyTorch!"

# Multilingual Synthesis
uv run python main.py --model chatterbox --text "Bonjour, comment ça va?" --language fr --multilingual
```

## ⚡ Chatterbox Turbo

For users prioritizing speed and low latency, a **Turbo** variant is available. It is ~2.5x faster than the standard model but is primarily optimized for English.

See the [Chatterbox Turbo Guide](CHATTERBOX_TURBO_GUIDE.md) for more details.

---

## 🛠️ Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--text` | string | required | Text to synthesize. |
| `--reference` | string | optional | Path to audio file for zero-shot cloning. |
| `--use-mlx` | flag | false | Use optimized MLX backend (Mac only). |
| `--multilingual` | flag | false | Use multilingual model (PyTorch only). |
| `--language` | string | `en` | Language code (e.g., `fr`, `es`, `zh`). |
| `--exaggeration` | float | `0.7` | Emotion intensity (0-1). |
| `--cfg-weight` | float | `0.5` | CFG guidance weight (0-1). Lower = faster pacing. |

---

## 💡 Best Practices
- **For Apple Silicon**: Always use `--use-mlx`. It provides a ~5-10x speedup over standard PyTorch on Mac.
- **For Voice Cloning**: Use a clear, 5-10 second reference clip for best results.
- **For Expressiveness**: Increase `--exaggeration` to `0.8+` for more dramatic speech.
