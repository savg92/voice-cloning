# Parakeet ASR - Usage Guide

**Parakeet TDT (Token-and-Duration Transducer)** is a highly efficient multilingual ASR model from NVIDIA, optimized for speed and accuracy. This project supports both the original NeMo implementation and an optimized Apple Silicon version via MLX.

## ⚠️ Important: Dependency Requirements

Parakeet's MLX backend (`mlx-audio`) requires **`transformers>=4.49.0`**, which conflicts with:
- **Chatterbox TTS** (requires `transformers==4.46.3`)

**Workaround**: Use separate virtual environments if you need both Parakeet and Chatterbox.

## Features

- **Multilingual**: Supports 100+ languages including Spanish, French, German, Chinese, Japanese, and many more
- **Blazing Fast**: TDT architecture allows for extremely fast inference
- **Dual Backend**:
  - **MLX**: Optimized for Apple Silicon (Mac M1/M2/M3)
  - **NeMo**: Optimized for NVIDIA GPUs
- **Timestamps**: Supported on MLX backend (SRT format)
- **Model**: `nvidia/parakeet-tdt-0.6b-v3` (600M parameters)

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

## Multilingual Support

Parakeet v3 supports 100+ languages with automatic language detection:

### Common Languages Supported

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | Spanish | es |
| French | fr | German | de |
| Chinese | zh | Japanese | ja |
| Korean | ko | Russian | ru |
| Arabic | ar | Portuguese | pt |
| Italian | it | Dutch | nl |

And many more...

### Usage with Non-English Audio

Parakeet automatically detects the language:

```bash
# Spanish audio
uv run python main.py --model parakeet \
  --reference samples/spanish_audio.wav \
  --output outputs/spanish_transcript.txt

# French audio
uv run python main.py --model parakeet \
  --reference samples/french_audio.wav \
  --output outputs/french_transcript.txt
```

## Troubleshooting

- **"MLX parakeet CLI not found"**: Ensure you are running on a Mac with Apple Silicon and have installed the dependencies.
- **"NeMo toolkit not installed"**: If on Linux/Windows/Intel Mac, you must install NeMo: `uv pip install "nemo_toolkit[asr]"`.
- **Empty Output**: Check if the audio file is valid and contains speech.

## Performance Notes

- **MLX**: Expect ~100x real-time performance on M2/M3 chips.
- **NeMo**: Expect state-of-the-art throughput on NVIDIA GPUs.
