# Parakeet ASR - Usage Guide

**Parakeet TDT (Token-and-Duration Transducer)** is a highly efficient ASR model from NVIDIA, optimized for speed and accuracy. This project supports both the original NeMo implementation and an optimized Apple Silicon version via MLX.

## Features

- **Blazing Fast**: TDT architecture allows for extremely fast inference.
- **Dual Backend**:
  - **MLX**: Optimized for Apple Silicon (Mac M1/M2/M3).
  - **NeMo**: Optimized for NVIDIA GPUs.
- **Timestamps**: Supported on MLX backend (SRT format).
- **English Only**: The current model (`parakeet-tdt-0.6b-v3`) is trained on English data.

## Installation

### MLX Backend (Apple Silicon)
Requires `uv` and `parakeet-mlx` CLI.
```bash
# Usually installed automatically via dependencies, or:
uv pip install mlx-audio
```

### NeMo Backend (NVIDIA GPU)
Requires NVIDIA NeMo toolkit.
```bash
uv pip install "nemo_toolkit[asr]"
```

## Usage Examples

### 1. Basic Transcription
```bash
uv run python main.py --model parakeet \
  --reference samples/speech.wav \
  --output outputs/transcript.txt
```

### 2. Transcription with Timestamps (MLX Only)
Generates an SRT file with timing information.
```bash
uv run python main.py --model parakeet \
  --reference samples/speech.wav \
  --timestamps \
  --output outputs/transcript.srt
```

## Troubleshooting

- **"MLX parakeet CLI not found"**: Ensure you are running on a Mac with Apple Silicon and have installed the dependencies.
- **"NeMo toolkit not installed"**: If on Linux/Windows/Intel Mac, you must install NeMo: `uv pip install "nemo_toolkit[asr]"`.
- **Empty Output**: Check if the audio file is valid and contains speech.

## Performance Notes

- **MLX**: Expect ~100x real-time performance on M2/M3 chips.
- **NeMo**: Expect state-of-the-art throughput on NVIDIA GPUs.
