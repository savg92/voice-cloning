# Dia2-1B TTS Model Guide

> [!CAUTION]
> **macOS Performance Warning**
> 
> Dia2-1B has **extremely poor performance on macOS**:
> - **MPS (Apple Silicon)**: Essentially unusable (0.0 toks/s, would take hours for short text)
> - **CPU**: Very slow (~840 seconds to generate 1.12s of audio)
> 
> **This model is CUDA-only for practical use.** Only recommended for Linux/Windows with NVIDIA GPUs.
> 
> For macOS users, consider alternative models: Kokoro, StyleTTS2, or Marvis instead.

## Overview

Dia2-1B is a streaming dialogue TTS model from Nari Labs capable of generating ultra-realistic conversational audio in real-time **on CUDA GPUs**. It can handle multi-speaker dialogues and supports voice cloning through prefix audio conditioning.

- **Model**: [nari-labs/Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B)
- **GitHub**: [nari-labs/dia2](https://github.com/nari-labs/dia2)
- **License**: Apache 2.0
- **⚠️ Platform**: **CUDA strongly recommended** (poor performance on CPU/MPS)

## Features

- **Streaming Generation**: Start generating audio before receiving the full text
- **Multi-Speaker Support**: Use `[S1]` and `[S2]` tags for dialogue
- **Voice Cloning**: Condition output on reference audio samples
- **Real-time Capable**: Optimized for low-latency speech-to-speech systems
- **CUDA Graph Support**: Accelerated inference on CUDA devices

## Installation

### 1. Install the dia2 library

The `dia2` library must be installed from GitHub:

```bash
uv pip install 'dia2 @ git+https://github.com/nari-labs/dia2.git'
uv pip install sphn whisper-timestamped
```

### 2. Verify installation

```bash
uv run python -c "from dia2 import Dia2; print('✓ dia2 installed')"
```

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.6.0+ (CUDA 12.6+ for GPU)
- **VRAM**: ~10GB recommended for optimal performance
- **Device Support**: CUDA (recommended), MPS, CPU

## Usage

### Command Line

#### Basic synthesis:

```bash
uv run python main.py \
  --model dia2 \
  --text "[S1] Hello! [S2] How are you today?" \
  --output outputs/output_dia2.wav
```

#### With custom parameters:

```bash
uv run python main.py \
  --model dia2 \
  --text "[S1] Good morning! [S2] Good morning to you too!" \
  --output outputs/output_dia2_conversation.wav \
  --cfg-scale 2.0 \
  --temperature 0.8 \
  --top-p 50
```

### Programmatic Usage

```python
from src.voice_cloning.tts.dia2 import Dia2TTS

# Initialize model
tts = Dia2TTS(
    model_name="nari-labs/Dia2-1B",
    device="cuda",  # or "mps", "cpu"
    dtype="bfloat16"
)

# Generate speech
audio = tts.synthesize(
    text="[S1] Hello Dia2!",
    output_path="outputs/output_dia2_hello.wav",
    cfg_scale=2.0,
    temperature=0.8,
    top_k=50,
    verbose=True
)
```

### Voice Cloning

Condition the generated speech on reference audio:

```python
audio = tts.synthesize(
    text="[S1] This should sound like the reference. [S2] Me too!",
    output_path="outputs/output_dia2_cloned.wav",
    prefix_speaker_1="reference_voice_1.wav",
    prefix_speaker_2="reference_voice_2.wav"
)
```

## Parameters

### Model Parameters

- `model_name` (str): HuggingFace model name
  - Default: `"nari-labs/Dia2-1B"`
  - Also available: `"nari-labs/Dia2-2B"` (larger model)
- `device` (str): Device to run on
  - Options: `"cuda"`, `"mps"`, `"cpu"`, or `None` (auto-detect)
- `dtype` (str): Model precision
  - Options: `"float32"`, `"float16"`, `"bfloat16"`
  - Default: `"bfloat16"`

### Generation Parameters

- `cfg_scale` (float): Classifier-free guidance scale
  - Range: 1.0 - 10.0
  - Default: 2.0
  - Higher values = more guided generation
- `temperature` (float): Sampling temperature
  - Range: 0.0 - 2.0
  - Default: 0.8
  - Lower = more deterministic, Higher = more creative
- `top_k` (int): Top-k sampling
  - Default: 50
  - Lower = more focused, Higher = more diverse
- `use_cuda_graph` (bool): Enable CUDA graph optimization
  - Default: `True`
  - Only works on CUDA devices

### Voice Cloning Parameters

- `prefix_speaker_1` (str): Path to reference audio for speaker 1
- `prefix_speaker_2` (str): Path to reference audio for speaker 2

## Text Format

Use speaker tags `[S1]` and `[S2]` to indicate different speakers:

```python
text = "[S1] Hello, how can I help you today? [S2] I'd like to know about your services."
```

## Performance

> [!WARNING]
> **macOS Users: This model is not practical on macOS devices.**

### Speed (Measured)

- **CUDA (NVIDIA GPU)**: Real-time or faster (~0.5x latency) ✅ **RECOMMENDED**
- **CPU (Apple Silicon M-series)**: ~750x slower than real-time ❌ **UNUSABLE**
  - Example: 840 seconds (14 minutes) to generate 1.12 seconds of audio
- **MPS (Apple Silicon GPU)**: 0.0 toks/s ❌ **COMPLETELY BROKEN**
  - Generation hangs indefinitely

### Quality

- **Naturalness**: Highly natural-sounding speech ⭐⭐⭐⭐⭐
- **Prosody**: Excellent intonation and rhythm ⭐⭐⭐⭐⭐
- **Dialogue**: Natural conversational flow ⭐⭐⭐⭐⭐
- **Voice Consistency**: Stable speaker characteristics ⭐⭐⭐⭐⭐

**Note**: Quality ratings based on CUDA performance. On CPU/MPS, generation is too slow to be practical.

## Troubleshooting

### Import Error

If you see `ModuleNotFoundError: No module named 'dia2'`:

```bash
uv pip install 'dia2 @ git+https://github.com/nari-labs/dia2.git'
uv pip install sphn whisper-timestamped
```

### CUDA Out of Memory

Reduce batch size or model size:

```python
# Use smaller model
tts = Dia2TTS(model_name="nari-labs/Dia2-1B", device="cuda")

# Or use CPU
tts = Dia2TTS(device="cpu")
```

### Slow Generation

Enable CUDA graph optimization (CUDA only):

```python
audio = tts.synthesize(
    text="[S1] Hello!",
    use_cuda_graph=True  # Faster on CUDA
)
```

## Differences from Other Models

| Feature | Dia2-1B | StyleTTS2 | Parler-TTS |
|---------|---------|-----------|------------|
| Multi-speaker | ✅ Yes | ❌ No | ❌ No |
| Streaming | ✅ Yes | ❌ No | ⚠️ Partial |
| Voice Cloning | ✅ Prefix | ✅ Reference | ❌ No |
| Speed | ⚡ Real-time | 🐢 Slow | ⚡ Fast |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## Examples

### Simple Greeting

```python
tts.synthesize(
    text="[S1] Good morning!",
    output_path="greeting.wav"
)
```

### Conversation

```python
tts.synthesize(
    text="""
    [S1] How was your day?
    [S2] It was great, thanks for asking!
    [S1] That's wonderful to hear.
    """,
    output_path="conversation.wav",
    cfg_scale=2.5
)
```

### Voice Cloning

```python
# Clone a specific voice for speaker 1
tts.synthesize(
    text="[S1] This will sound like the reference voice.",
    output_path="cloned_voice.wav",
    prefix_speaker_1="my_voice_sample.wav"
)
```

## References

- [Dia2 GitHub Repository](https://github.com/nari-labs/dia2)
- [Dia2-1B Model Card](https://huggingface.co/nari-labs/Dia2-1B)
- [Dia2-2B Model Card](https://huggingface.co/nari-labs/Dia2-2B)
- [Nari Labs Website](https://narilabs.org)
