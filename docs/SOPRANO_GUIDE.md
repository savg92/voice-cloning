# Soprano-1.1-80M Guide

Soprano is an ultra-lightweight (80M parameters) text-to-speech model optimized for speed and naturalness. It achieves extremely high inference speeds, making it ideal for real-time applications.

## Features

- **Architecture**: Causal language model based on Qwen3.
- **Language**: English Only.
- **Speed**: Up to 2000x real-time inference.
- **Sample Rate**: 32kHz.
- **Streaming**: Supports smooth, real-time streaming on both PyTorch and MLX backends.
- **Backends**: Supports both standard PyTorch and optimized MLX for Apple Silicon.

## Usage

### Web UI

1.  Launch the Web UI: `uv run main.py --model web`
2.  Select **Soprano** from the "Model Engine" dropdown.
3.  Enter your text and click **Generate Audio**.

#### New Settings:
- **MLX Acceleration**: Enable this for macOS (Apple Silicon). Uses the optimized `mlx-audio` backend.
- **Quality Controls**:
    - **Temperature**: Controls randomness. Higher (e.g., 0.7-0.8) sounds more natural/varied. Lower (0.0-0.3) sounds more stable.
    - **Top P**: Nucleus sampling parameter (Default: 0.95).

### CLI

Synthesize speech using the CLI:

```bash
uv run main.py --model soprano --text "Hello, I am Soprano." --output output.wav
```

Advanced parameters:

```bash
uv run main.py --model soprano --text "Hello." --temperature 0.8 --top-p 0.9
```

To use the MLX backend:

```bash
uv run main.py --model soprano --text "Hello, I am Soprano." --output output.wav --use-mlx
```

Enable streaming playback:

```bash
uv run main.py --model soprano --text "Streaming test." --stream
```

## Dependencies

- `soprano-tts` (PyTorch)
- `mlx-audio` (MLX)
- `sounddevice` (Streaming playback)

## Troubleshooting

- **Robotic Sound**: Try increasing the **Temperature** to 0.7 or 0.8.
- **Streaming Issues**: Ensure you have `sounddevice` installed. On macOS, this usually works out of the box with `uv`. Streaming is supported on both **Torch** and **MLX** backends.