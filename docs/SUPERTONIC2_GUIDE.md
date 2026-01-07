# Supertonic-2 TTS - Usage Guide

**Supertone Supertonic-2** is an optimized, multilingual, on-device TTS system using ONNX Runtime. It is the successor to the original Supertonic model, adding support for 5 languages and improved quality while maintaining ultra-fast performance.

## Features

- ⚡ **Ultra Fast**: Optimized for on-device inference using ONNX Runtime.
- 🌍 **Multilingual**: Supports English (EN), Korean (KO), Spanish (ES), Portuguese (PT), and French (FR).
- 🪶 **Lightweight**: Small footprint suitable for mobile and edge devices.
- 📱 **Private**: All processing happens locally on your device.
- 🔄 **Auto-Download**: Models are automatically downloaded from Hugging Face on first use.

## Installation

### Prerequisites

1. **Install onnxruntime**:
   ```bash
   uv pip install onnxruntime
   ```

2. **Models**:
   The toolkit will automatically download models to `models/supertonic2` when you first run the model.

## Usage Examples

### 1. Basic Synthesis (English)
```bash
uv run python main.py --model supertonic2 \
  --text "Hello, this is Supertonic-2 TTS." \
  --output outputs/supertonic2_en.wav
```

### 2. Korean Synthesis
```bash
uv run python main.py --model supertonic2 \
  --text "안녕하세요, 슈퍼토닉-2 모델입니다." \
  --language ko \
  --output outputs/supertonic2_ko.wav
```

### 3. Change Voice Style
Available voice presets: **F1, F2** (female), **M1, M2** (male)
```bash
uv run python main.py --model supertonic2 \
  --text "This is a different voice preset." \
  --voice M1 \
  --output outputs/supertonic2_m1.wav
```

### 4. Adjust Quality (Inference Steps)
Default is 10. Higher steps can improve quality but slightly increase latency.
```bash
uv run python main.py --model supertonic2 \
  --text "Synthesizing with more steps." \
  --steps 20 \
  --output outputs/supertonic2_high.wav
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--language` | str | en | Language code (en, ko, es, pt, fr) |
| `--voice` | str | F1 | Voice preset (F1, F2, M1, M2) |
| `--steps` | int | 10 | Inference steps |
| `--speed` | float | 1.0 | Speech speed multiplier |

## Troubleshooting

- **"Model directory not found"**: The model should download automatically. Ensure you have an internet connection for the first run.
- **"onnxruntime not found"**: `uv pip install onnxruntime`

## Performance

- **Apple Silicon**: Leverages CoreML/MPS for acceleration via ONNX Runtime providers.
- **Latency**: Aiming for < 0.1 RTF on modern hardware.
- **Output**: 16-bit WAV at 24kHz.
