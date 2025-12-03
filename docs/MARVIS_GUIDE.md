# Marvis TTS - Advanced Features Guide

This guide explains how to use Marvis TTS's advanced features including **streaming** and **voice cloning**.

## Basic Usage

```bash
uv run python main.py --model marvis --text "Hello world" --output outputs/marvis_basic.wav
```

## Multilingual Support

Marvis v0.2 supports **English, French, and German**:

```bash
# French
uv run python main.py --model marvis \
  --text "Bonjour le monde" \
  --output outputs/marvis_french.wav

# German
uv run python main.py --model marvis \
  --text "Guten Tag Welt" \
  --output outputs/marvis_german.wav
```

> **Note**: Language is automatically detected from the input text. For best results with voice cloning in non-English languages, ensure your reference audio matches the target language.

## Voice Cloning

Clone a voice by providing a reference audio file. Marvis will automatically transcribe the reference audio and clone its voice characteristics.

### Basic Voice Cloning

```bash
uv run python main.py --model marvis \
  --text "This is a test of voice cloning" \
  --reference samples/anger.wav \
  --output outputs/marvis_cloned.wav
```

### Voice Cloning with Manual Transcription

For better accuracy, you can provide the text caption for the reference audio:

```bash
uv run python main.py --model marvis \
  --text "This is a test of voice cloning" \
  --reference samples/anger.wav \
  --ref_text "The actual text spoken in the reference audio" \
  --output outputs/marvis_cloned.wav
```

## Streaming Mode

Enable real-time audio playback as the model generates speech (audio plays as it's being synthesized):

```bash
uv run python main.py --model marvis \
  --text "This will stream as it generates" \
  --stream \
  --output outputs/marvis_stream.wav
```

## Speed Control

Adjust the speech speed:

```bash
# Slower speech (0.5x speed)
uv run python main.py --model marvis \
  --text "Speaking slowly" \
  --speed 0.5 \
  --output outputs/marvis_slow.wav

# Faster speech (1.5x speed)
uv run python main.py --model marvis \
  --text "Speaking quickly" \
  --speed 1.5 \
  --output outputs/marvis_fast.wav
```

## Temperature Control

Control the randomness of the generation (higher = more varied, lower = more conservative):

```bash
uv run python main.py --model marvis \
  --text "Testing temperature control" \
  --temperature 0.9 \
  --output outputs/marvis_temp.wav
```

## Combined Features

You can combine multiple features:

```bash
uv run python main.py --model marvis \
  --text "Clone my voice with streaming at custom speed" \
  --reference samples/my_voice.wav \
  --ref_text "Original text from my voice sample" \
  --stream \
  --speed 1.2 \
  --temperature 0.7 \
  --output outputs/marvis_advanced.wav
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--text` | string | *required* | Text to synthesize |
| `--output` | string | *required* | Output audio file path |
| `--reference` | string | optional | Reference audio for voice cloning |
| `--ref_text` | string | optional | Text caption for reference audio |
| `--stream` | flag | false | Enable streaming playback |
| `--speed` | float | 1.0 | Speech speed multiplier |
| `--temperature` | `float` | `0.7` | Sampling temperature (0.0-1.0). Higher = more random/creative. |
| `--quantized` | `flag` | `True` | Use 4-bit quantized model for faster speed (default). |
| `--no-quantized`| `flag` | `False` | Use full precision model (slower). |
## Tips

1. **Voice Cloning**: Works best with clear, high-quality reference audio (3-10 seconds)
2. **Streaming**: Useful for real-time applications, but file is still saved
3. **Speed**: Keep between 0.5-2.0 for natural-sounding speech
4. **Temperature**: 0.5-0.8 works best; higher values may add artifacts

## Known Issues

> [!WARNING]
> There is currently a bug in the `mlx-audio` library with the Marvis v0.2 model (`KeyError: 'rope_scaling'`). This affects all Marvis features including basic synthesis, voice cloning, and streaming. 
>
> **Workaround**: Use alternative TTS models (Kokoro, Kitten TTS, or Chatterbox) until this is resolved.
>
> The implementation is complete and will work once the upstream bug is fixed.
