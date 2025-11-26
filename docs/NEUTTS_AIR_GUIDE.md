# NeuTTS Air - Usage Guide

**NeuTTS Air** is a voice cloning TTS model that creates ultra-realistic voices with just 3 seconds of reference audio.

## Features

- ⚡ **On-Device**: Optimized for phones, laptops, Raspberry Pi
- 🗣 **Voice Cloning**: Clone any voice with 3+ seconds of audio
- 🚀 **Fast**: Built on 0.5B Qwen backbone
- 🎯 **High Quality**: Uses NeuCodec for natural speech
- 🌍 **Language**: English only (trained on "opensource-en" dataset)

## Language Support

**NeuTTS Air currently supports English only.**

- Trained on English dataset
- Best results with English reference audio and text
- Accent characteristics from reference may be preserved
- For multilingual TTS, use Kokoro or Chatterbox from this toolkit

## Installation

### System Requirements
- **Python**: 3.11+ (3.12 recommended)
- **OS**: macOS, Linux, or Windows
- **RAM**: 4GB+ recommended
- **Storage**: ~5GB for models (first run)

### Step 1: Install System Dependencies

**macOS:**
```bash
brew install espeak-ng
```

**Ubuntu/Debian:**
```bash
sudo apt install espeak-ng
```

**Windows:**
Download and install espeak-ng from [official repository](https://github.com/espeak-ng/espeak-ng/releases)

### Step 2: Install Python Dependencies

The dependencies are automatically installed when you install the project:
```bash
cd voice-cloning
uv pip install -e .
```

This installs:
- `neucodec` - Audio codec
- `phonemizer` - Text-to-phoneme conversion
- `resemble-perth` - Audio processing
- `llama-cpp-python` - GGUF model support

### Step 3: Verify Installation

Test that everything is working:
```bash
uv run python -c "
import sys
sys.path.insert(0, 'models')
from neuttsair.neutts import NeuTTSAir
print('✓ NeuTTS Air ready!')
"
```

### Step 4: First Run (Downloads Models)

The first time you run NeuTTS Air, it will automatically download the models (~2-4GB):

```bash
uv run python main.py --model neutts-air \
  --text "Hello, this is my first test!" \
  --reference samples/neutts_air/dave.wav \
  --output outputs/first_test.wav
```

**Expected behavior:**
- ⏳ Downloads backbone model (neuphonic/neutts-air-q4-gguf) - ~2GB
- ⏳ Downloads codec model (neuphonic/neucodec) - ~500MB
- ⏳ First run takes 5-10 minutes
- ✓ Subsequent runs are much faster (models cached)

### Notes
- Models are cached in `~/.cache/huggingface/`
- Debug logging shows progress during initialization
- espeak-ng is auto-configured via environment variables

## Usage

### Basic Voice Cloning
```bash
uv run python main.py --model neutts-air \
  --text "Hello, this is a cloned voice!" \
  --reference samples/neutts_air/dave.wav \
  --ref-text samples/neutts_air/dave.txt \
  --output outputs/cloned_neutts_air.wav
```

### Auto-detect Reference Text
If your reference text file has the same name as the audio:
```bash
uv run python main.py --model neutts-air \
  --text "The reference text will be auto-detected." \
  --reference samples/neutts_air/dave.wav \
  --output outputs/output_neutts_air.wav
```
(looks for `dave.txt` automatically)

### Different Backbone Models
```bash
# Q4 quantized (default, faster)
--backbone neuphonic/neutts-air-q4-gguf

# Q8 quantized (better quality)
--backbone neuphonic/neutts-air-q8-gguf
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--reference` | str | Required | Reference audio file (3+ seconds) |
| `--ref-text` | str | Optional | Reference text transcript |
| `--backbone` | str | neutts-air-q4-gguf | Backbone model |

## Tips for Best Results

1. **Reference Audio Quality**:
   - Clean, clear audio
   - Minimal background noise
   - At least 3 seconds long
   - Single speaker

2. **Reference Text**:
   - Must match the reference audio transcript
   - Include punctuation
   - Maintain natural phrasing

3. **Input Text**:
   - Works best with natural language
   - Handles punctuation well
   - Supports various speaking styles

## Output

- **Format**: WAV
- **Sample Rate**: 24kHz
- **Quality**: High-fidelity voice cloning

## Available Backbones

From [NeuTTS Air Collection](https://huggingface.co/collections/neuphonic/neutts-air-68cc14b7033b4c56197ef350):
- `neuphonic/neutts-air-q4-gguf` (default)
- `neuphonic/neutts-air-q8-gguf`

## Troubleshooting

- **"Reference text not found"**: Create a `.txt` file with the same name as your audio
- **"neucodec not found"**: Run `uv pip install neucodec`
- **"espeak not installed"**: The phonemizer library looks for `espeak` but you have `espeak-ng`. Create a symlink:
  ```bash
  # macOS
  sudo ln -s /opt/homebrew/bin/espeak-ng /usr/local/bin/espeak
  ```
