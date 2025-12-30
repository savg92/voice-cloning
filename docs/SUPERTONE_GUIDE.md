# Supertone (Supertonic) TTS - Usage Guide

**Supertone Supertonic** is an ultra-fast, lightweight TTS model designed for on-device inference using ONNX Runtime. It generates speech **167× faster than real-time** on modern hardware.

## Features

- ⚡ **Lightning Fast**: 167× real-time on M4 Pro, fastest TTS available
- 🪶 **Ultra Lightweight**: Only 66M parameters
- 📱 **On-Device**: Complete privacy, zero latency
- ⚙️ **Configurable**: Adjustable inference steps
- 🎨 **Natural Text Handling**: Handles numbers, dates, abbreviations

## Installation

### Prerequisites

1. **Install Git LFS** (required for downloading large model files):
   ```bash
   # macOS
   brew install git-lfs && git lfs install
   
   # Linux (Debian/Ubuntu)
   sudo apt-get install git-lfs && git lfs install
   ```

2. **Install onnxruntime**:
   ```bash
   uv pip install onnxruntime
   ```

3. **Download Models**:
   ```bash
   mkdir -p models
   git clone https://huggingface.co/Supertone/supertonic models/supertonic
   ```

## Usage Examples

### 1. Basic Synthesis (Default: Female voice F1)
```bash
uv run python main.py --model supertone \
  --text "Hello, this is Supertone TTS." \
  --output outputs/supertone.wav
```

### 2. Use a Different Voice
Available voice presets: **F1, F2** (female), **M1, M2** (male)
```bash
uv run python main.py --model supertone \
  --text "This is a male voice." \
  --preset M1 \
  --output outputs/male_voice.wav
```

### 3. Adjust Quality (Inference Steps)
Higher steps = better quality but slower. Default is 8.
```bash
uv run python main.py --model supertone \
  --text "High quality speech with 16 steps." \
  --steps 16 \
  --output outputs/high_quality.wav
```

### 4. Streaming Playback
Enable pseudo-streaming to play sentences as they are generated.
```bash
uv run python main.py --model supertone \
  --text "This is sentence one. This is sentence two. This is sentence three." \
  --stream \
  --output outputs/streamed.wav
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--preset` | str | F1 | Voice preset (F1, F2, M1, M2) |
| `--steps` | int | 8 | Inference steps (higher = better quality) |
| `--stream` | flag | False | Enable streaming playback |

**Available Voice Presets:**
- **F1**: Female voice 1 (default)
- **F2**: Female voice 2
- **M1**: Male voice 1
- **M2**: Male voice 2

## Troubleshooting

- **"Model directory not found"**: Run the git clone command above to download models.
- **"Git LFS not installed"**: Install Git LFS and run `git lfs pull` in the models directory.
- **"onnxruntime not found"**: `uv pip install onnxruntime`

## Performance

- **M4 Pro**: ~167× real-time
- **CPU-only**: Still extremely fast compared to other TTS systems
- **Output**: 16-bit WAV at 24kHz
