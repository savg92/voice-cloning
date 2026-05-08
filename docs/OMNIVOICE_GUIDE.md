# OmniVoice Guide

[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) is a massively multilingual zero-shot text-to-speech (TTS) model developed by the **k2-fsa** team. It supports over 600 languages and provides state-of-the-art voice cloning and voice design capabilities.

## Features

- **Massively Multilingual**: Supports 600+ languages.
- **Zero-shot Voice Cloning**: Clone any voice from a 3-10 second audio clip.
- **Voice Design**: Control voice attributes using natural language descriptions.
- **Fast Inference**: High-speed synthesis on both GPU and CPU.

## Installation

OmniVoice requires specific versions of `torch` and `transformers` (v5+).

```bash
uv add omnivoice
```

## Usage

### 1. Basic TTS

Generate speech in a default/random voice.

```bash
uv run python main.py --model omnivoice --text "Hello, I am speaking with OmniVoice."
```

### 2. Voice Cloning

Clone a voice using a reference audio sample.

```bash
uv run python main.py --model omnivoice \
  --text "I am speaking with your cloned voice!" \
  --reference samples/ref.wav
```

You can optionally provide the transcript of the reference audio for better quality:

```bash
uv run python main.py --model omnivoice \
  --text "I am speaking with your cloned voice!" \
  --reference samples/ref.wav \
  --ref_text "The quick brown fox jumps over the lazy dog."
```

### 3. Voice Design (Instruction-based)

Create a custom voice using specific descriptors (gender, age, pitch, accent, style).

Valid descriptors include: `male, female, child, teenager, young adult, middle-aged, elderly, low pitch, high pitch, american accent, british accent, whisper`, etc.

```bash
uv run python main.py --model omnivoice \
  --text "This is a custom voice designed by AI." \
  --instruct "female, young adult, british accent, high pitch"
```

### Troubleshooting

#### 1. libtorchcodec / FFmpeg Issues (Mac)
If you see errors about `libtorchcodec` failing to load, ensure FFmpeg is installed via Homebrew:
```bash
brew install ffmpeg
```

#### 2. Out of Memory (OOM) / SIGKILL (Exit Code 137)
OmniVoice + Whisper (for cloning) requires significant memory (~6-8GB+). On 8GB Macs, the toolkit now applies these optimizations automatically:
- **Automatic Light ASR**: If `--ref_text` is missing, the toolkit uses `faster-whisper` (tiny) on CPU to transcribe the reference audio *before* loading OmniVoice. This saves ~1.5GB of RAM compared to the model's internal ASR.
- **Reference Path Handling**: Reference audio is passed as a file path rather than a pre-loaded tensor to further reduce memory pressure.

**Tips for better stability:**
- **Provide `--ref_text`**: Manually providing the transcript is still the most stable way to ensure success.
- **Close other apps**: Ensure plenty of unified memory is available.
- **Use `--device mps`**: GPU memory management is generally more efficient for these models.

#### 3. Voice Design Items
Remember that `--instruct` requires comma-separated keywords from the pre-defined list, not free-form natural language.

### 4. Multilingual Synthesis

Specify the language (if not auto-detected or if you want to force it).

```bash
uv run python main.py --model omnivoice \
  --text "Bonjour, comment allez-vous?" \
  --language fr
```

## Advanced Options

- `--speed`: Adjust speech speed (e.g., `--speed 1.2`)
- `--device`: Force a specific device (`cuda`, `mps`, `cpu`)

## Technical Details

- **Model ID**: `k2-fsa/OmniVoice`
- **Architecture**: Diffusion-based language model
- **Sampling Rate**: 24,000 Hz
