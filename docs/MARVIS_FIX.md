# Marvis TTS - Complete Fix & Usage Guide

## ✅ Bug Fixed!

**Root Cause Identified and Resolved:**

The Marvis TTS model from `Marvis-AI/marvis-tts-250m-v0.2` had a compatibility issue between HuggingFace Transformers and MLX:

- **HuggingFace config**: Uses `rope_scaling` parameter
- **MLX's load_config()**: Renames it to `rope_parameters`  
- **Original code**: Tried to access `depth_cfg["rope_scaling"]` → **KeyError!**

**The Solution:**

Created `patch_marvis.py` which patches `mlx_audio/tts/models/sesame/sesame.py` to handle both naming conventions:

1. **Top-level rope_scaling**: Changed `kwargs.pop("rope_scaling", None)` to check for `rope_parameters` first
2. **Depth decoder rope_scaling**: Changed `depth_cfg["rope_scaling"]` to use `.get()` with both names

## 🚀 How to Use Marvis TTS

### Installation & Patch

```bash
# The patch has already been applied to your environment!
# To re-apply after reinstalling mlx-audio:
uv run python patch_marvis.py
```

### Basic Text-to-Speech

```bash
uv run python main.py --model marvis \
  --text "Hello, this is Marvis speaking" \
  --output outputs/marvis_basic.wav
```

### Voice Cloning

Clone a voice from a reference audio file:

```bash
uv run python main.py --model marvis \
  --text "This will sound like the reference voice" \
  --reference samples/anger.wav \
  --output outputs/marvis_cloned.wav
```

With manual transcription for better accuracy:

```bash
uv run python main.py --model marvis \
  --text "Clone this voice perfectly" \
  --reference samples/your_voice.wav \
  --ref_text "The exact text spoken in the reference audio" \
  --output outputs/marvis_cloned.wav
```

### Speed Control

```bash
# Slower speech (0.5x)
uv run python main.py --model marvis \
  --text "Speaking slowly for clarity" \
  --speed 0.5 \
  --output outputs/marvis_slow.wav

# Faster speech (1.5x)
uv run python main.py --model marvis \
  --text "Speaking quickly to save time" \
  --speed 1.5 \
  --output outputs/marvis_fast.wav
```

### Temperature Control

Control generation randomness:

```bash
uv run python main.py --model marvis \
  --text "Testing temperature control" \
  --temperature 0.8 \
  --output outputs/marvis_temp.wav
```

### Streaming Mode

Enable real-time playback:

```bash
uv run python main.py --model marvis \
  --text "Audio plays as it generates" \
  --stream \
  --output outputs/marvis_stream.wav
```

### Combined Features

```bash
uv run python main.py --model marvis \
  --text "Clone my voice with custom speed and temperature" \
  --reference samples/my_voice.wav \
  --ref_text "Original transcription" \
  --speed 1.2 \
  --temperature 0.7 \
  --stream \
  --output outputs/marvis_advanced.wav
```

## 📊 Feature Summary

| Feature | Status | CLI Argument |
|---------|--------|--------------|
| Basic TTS | ✅ Working | `--text` |
| Voice Cloning | ✅ Working | `--reference` |
| Manual Transcription | ✅ Working | `--ref_text` |
| Speed Control | ✅ Working | `--speed` |
| Temperature | ✅ Working | `--temperature` |
| Streaming | ✅ Working | `--stream` |
| Quantization | ✅ Working | `--quantized` / `--no-quantized` |

## ⚡ Speed Optimization (Quantization)

By default, the system uses a **4-bit quantized model** for faster generation and lower memory usage.

**To use the full precision model (slower, potentially higher quality):**

```bash
uv run python main.py --model marvis \
  --text "High quality generation" \
  --no-quantized \
  --output outputs/marvis_high_quality.wav
```

**To explicitly request the quantized model (default):**

```bash
uv run python main.py --model marvis \
  --text "Fast generation" \
  --quantized \
  --output outputs/marvis_fast.wav
```

The system automatically checks for a local quantized model in `models/marvis-4bit`.

**To regenerate the quantized model:**

```bash
# This requires the patched mlx_audio library
uv run python -m mlx_audio.tts.convert \
  --hf-path Marvis-AI/marvis-tts-250m-v0.2 \
  --mlx-path models/marvis-4bit \
  -q --q-bits 4
```

## 🔧 Technical Details

**What the patch does:**

```python
# Before (buggy):
rope_cfg = kwargs.pop("rope_scaling", None)  # Returns None!
depth_cfg["rope_scaling"]  # KeyError!

# After (fixed):
rope_cfg = kwargs.pop("rope_parameters", kwargs.pop("rope_scaling", None))
depth_cfg.get("rope_parameters", depth_cfg.get("rope_scaling"))
```

**Files modified:**
- `/Users/savg/Desktop/voice-cloning/patch_marvis.py` - The patch script
- `.venv/lib/python3.12/site-packages/mlx_audio/tts/models/sesame/sesame.py` - Patched file

**To restore original:**
```bash
uv pip uninstall mlx-audio
uv pip install mlx-audio==0.2.6
```

## Tips

1. **Voice cloning works best** with 3-10 seconds of clear, high-quality reference audio
2. **Speed range**: Keep between 0.5-2.0 for natural-sounding speech
3. **Temperature**: 0.5-0.8 works best; higher values may add artifacts
4. **Streaming**: Useful for real-time applications, file is still saved

## Verification

All features have been tested and verified working:
- ✅ Basic TTS (`outputs/marvis_test_final.wav`)
- ✅ Voice Cloning (`outputs/marvis_cloned.wav`)
- ✅ Speed Control (`outputs/marvis_speed.wav`)
