# KittenTTS Nano - Usage Guide

**KittenTTS Nano** is a lightweight, CPU-friendly text-to-speech model designed for speed and efficiency. It supports multiple expressive voices and speed control. **Currently English-only** (v0.1 and v0.2).

## Features

- **Lightweight**: Runs efficiently on CPU.
- **Multiple Voices**: 8 expressive voices (male/female).
- **Speed Control**: Adjust speech rate.
- **Versions**: Supports v0.1 and v0.2 models.
- **Language**: **English only** (multilingual support planned for future releases).

> [!NOTE]
> For multilingual TTS, consider using **Kokoro** which supports English, French, Japanese, Chinese, Spanish, Italian, Portuguese, and Hindi.

## Usage Examples

### 1. Basic Synthesis (Default Voice)
```bash
uv run python main.py --model kitten \
  --text "Hello, this is Kitten TTS." \
  --output outputs/kitten.wav
```

### 2. Change Voice
Use the `--voice` argument to select a specific voice.
```bash
uv run python main.py --model kitten \
  --text "This is a different voice." \
  --voice expr-voice-3-m \
  --output outputs/kitten_male.wav
```

### 3. Adjust Speed
Use `--speed` to change the speaking rate (default: 1.0).
```bash
uv run python main.py --model kitten \
  --text "Speaking very quickly now." \
  --speed 1.5 \
  --output outputs/kitten_fast.wav
```

### 4. Streaming Playback
Enable pseudo-streaming to play sentences as they are generated.
```bash
uv run python main.py --model kitten \
  --text "This is sentence one. This is sentence two. This is sentence three." \
  --stream \
  --output outputs/kitten_stream.wav
```

### 4. Specific Model Version
You can explicitly select v0.1 or v0.2 (default is v0.2).
```bash
uv run python main.py --model kitten-0.1 \
  --text "Using version 0.1." \
  --output outputs/kitten_v1.wav
```

## Available Voices

| Voice ID | Gender | Description |
|----------|--------|-------------|
| `expr-voice-2-f` | Female | Expressive |
| `expr-voice-2-fm`| Female | Expressive (Mix?) |
| `expr-voice-3-f` | Female | Expressive |
| `expr-voice-3-m` | Male | Expressive |
| `expr-voice-4-f` | Female | **Default** |
| `expr-voice-4-m` | Male | Expressive |
| `expr-voice-5-f` | Female | Expressive |
| `expr-voice-5-m` | Male | Expressive |

## Troubleshooting

- **"Failed to import kittentts"**: Ensure the package is installed (`uv pip install kittentts`).
- **"Espeak not found"**: On macOS, you might need `brew install espeak-ng`. The wrapper attempts to patch paths automatically.
