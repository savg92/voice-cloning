# CosyVoice2 Guide

This guide explains how to use the **CosyVoice2** model (0.5B) for high-quality text-to-speech, zero-shot voice cloning, and style control.

We support two backends:
1. **MLX (Recommended for Mac)**: Optimized for Apple Silicon.
2. **PyTorch (Standard)**: Standard implementation (slower on Mac).

## Quick Start (CLI)

The easiest way to use CosyVoice2 is via the Command Line Interface (CLI) using `uv`.

### Basic Synthesis
Generate speech from text using the optimized MLX backend:

```bash
uv run python main.py --model cosyvoice --text "Hello, I am speaking from the command line." --use-mlx
```

### Zero-Shot Voice Cloning
Clone a voice from a reference audio file:

```bash
uv run python main.py \
  --model cosyvoice \
  --text "This is a cloned voice." \
  --reference path/to/reference_audio.wav \
  --output cloned_output.wav \
  --use-mlx
```

### Instruct Mode (Style Control)
Control the emotion or style using `--emotion`:

```bash
uv run python main.py \
  --model cosyvoice \
  --text "I am speaking with a specific emotion." \
  --reference path/to/reference.wav \
  --emotion "Speak in a sad and slow tone" \
  --use-mlx
```

## Installation

1. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **PyTorch Backend Setup** (Optional):
   For the PyTorch backend, you need the official CosyVoice repository in `models/`:
   ```bash
   mkdir -p models
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git models/CosyVoice
   cd models/CosyVoice
   git submodule update --init --recursive
   # UV installation of CosyVoice deps (relaxed where possible)
   uv pip install -r requirements.txt
   cd ../..
   ```

3. **Verify Installation**:
   ```bash
   uv run python tests/verify_cosyvoice.py
   ```

## Advanced Usage (Python API)

You can also use CosyVoice2 directly in your Python scripts.

```python
from voice_cloning.tts.cosyvoice import synthesize_speech

# Basic Typography
synthesize_speech(
    text="Hello world",
    output_path="output.wav",
    use_mlx=True
)

# Cloning
synthesize_speech(
    text="Cloned voice",
    ref_audio_path="reference.wav",
    use_mlx=True
)
```

## Troubleshooting

- **PyTorch Backend Issues**: Ensure `CosyVoice` is cloned into `models/CosyVoice` and dependencies are installed.
- **Microphone Access**: If recording reference audio, ensure terminal has microphone permissions.
- **Reference Audio**: For best cloning results, use 5-10 seconds of clear speech (24kHz+ recommended).
