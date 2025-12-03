# Voice Cloning Project

This project provides a comprehensive testing and comparison platform for multiple text-to-speech (TTS) models: Chatterbox, Kitten TTS Nano, Kokoro, Marvis TTS, Parakeet ASR, Canary ASR, Granite ASR, and Whisper ASR. It enables easy voice cloning and synthesis with different models to evaluate their performance and quality.

## Features

- **Multiple TTS Models**: Support for Chatterbox, Kitten TTS Nano, Kokoro, Marvis TTS, Supertone, NeuTTS Air, Dia2-1B (CUDA only)
- **Multiple ASR Models**: Parakeet, Canary, Granite, Whisper
- **VAD**: HumAware-VAD
- **Voice Cloning**: Clone voices using reference audio samples. Models like Chatterbox and Marvis TTS support voice cloning.
- **Multi-language Support**: Models like Kokoro, Marvis TTS, and Parakeet support multiple languages.
- **Command-line Interface**: Easy-to-use CLI for testing different models
- **Flexible Configuration**: Customizable parameters for each model

## Installation & Quick Start

### Prerequisites
- **Python 3.11+** (Python 3.12 recommended)
- **uv** (fast Python package manager)
- **System Dependencies**:
  - macOS: `brew install espeak-ng ffmpeg git-lfs`
  - Ubuntu/Debian: `sudo apt install espeak-ng ffmpeg git-lfs`

### 1. Install uv (if not already installed)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/savg92/voice-cloning.git
cd voice-cloning
uv venv
uv pip install -e .
```

### 3. Quick Start Examples

#### TTS (Text-to-Speech)
```bash
# Kokoro - High quality, multilingual
uv run python main.py --model kokoro \
  --text "Hello, this is Kokoro TTS!" \
  --output outputs/kokoro.wav

# NeuTTS Air - Voice cloning (requires reference audio)
uv run python main.py --model neutts-air \
  --text "Your cloned voice here!" \
  --reference samples/neutts_air/dave.wav \
  --output outputs/cloned.wav

# Supertone - Ultra-fast (requires model download)
git clone https://huggingface.co/Supertone/supertonic models/supertonic
uv run python main.py --model supertone \
  --text "Super fast synthesis!" \
  --output outputs/supertone.wav

# Dia2-1B - Multi-speaker dialogue (⚠️ CUDA only, slow on macOS)
uv run python main.py --model dia2 \
  --device cpu \
  --text "[S1] Hello! [S2] How are you?" \
  --output outputs/dialogue.wav
```

#### ASR (Speech Recognition)
```bash
# Whisper - General purpose, multilingual
uv run python main.py --model whisper \
  --reference path/to/audio.wav \
  --output transcript.txt

# Parakeet - Fast, English
uv run python main.py --model parakeet \
  --reference path/to/audio.wav \
  --output transcript.txt
```

#### VAD (Voice Activity Detection)
```bash
# HumAware - Distinguish speech from humming
uv run python main.py --model humaware \
  --reference path/to/audio.wav \
  --output segments.json
```

### 4. Model-Specific Setup

Some models require additional setup:

- **Supertone**: Download models via `git clone https://huggingface.co/Supertone/supertonic models/supertonic`
- **NeuTTS Air**: Requires `llama-cpp-python` (auto-installed)
- **Dia2-1B**: Requires manual installation from GitHub - see [docs/DIA2_GUIDE.md](docs/DIA2_GUIDE.md). **CUDA-only recommended.**
- **Chatterbox**: Has dependency conflicts - see [docs/CHATTERBOX_GUIDE.md](docs/CHATTERBOX_GUIDE.md)

For detailed usage of each model, see the respective guide in `docs/`.

## Quick Reference

### 🎭 Voice Cloning Models
- **Chatterbox**: 23 languages, best for multilingual cloning
- **Marvis**: English/French/German, streaming support
- **NeuTTS Air**: English only, fastest cloning (3s+ reference)
- **Dia2**: Multi-speaker dialogue (CUDA only)

### 🌍 Multilingual Models
- **TTS**: Chatterbox (23 langs), Kokoro (8 langs), Marvis (EN/FR/DE)
- **ASR**: Parakeet (100+ langs), Whisper (99+ langs), Canary (25 langs)

### 🍎 Apple Silicon Optimized (MLX)
- **Kokoro** (`--use-mlx`): 30% faster TTS
- **Marvis**: Native MLX, streaming + voice cloning
- **Parakeet**: MLX backend for fastest ASR

## Supported Models

### 1. Chatterbox (Voice Cloning) ⚠️
### 1. Chatterbox TTS 🎭 Voice Cloning | ⚠️ Dependency Conflict
- **Type**: Zero-shot TTS (Encoder-Decoder)
- **Languages**: 23 languages including English, Spanish, French, German, Arabic, Chinese, Japanese, Korean
- **Features**: 
  - ✨ **Voice cloning** with reference audio
  - Exaggeration/emotion control
  - Multilingual support
  - See [docs/CHATTERBOX_GUIDE.md](docs/CHATTERBOX_GUIDE.md) for details.

### 2. Kitten TTS Nano (Lightweight TTS)
- **Type**: Fast, CPU-friendly TTS
- **Features**: 
  - 8 Expressive voices
  - Speed control
  - **Streaming**: Supports pseudo-streaming playback (`--stream`)
  - See [docs/KITTEN_GUIDE.md](docs/KITTEN_GUIDE.md) for usage details.
- **Best for**: Real-time applications, low-resource devices

### 3. Kokoro (High Quality TTS)
- **Type**: Neural TTS (82M params)
- **Features**: 
  - High quality, natural sounding
  - Multilingual (En, Fr, Jp, Zh, Es, It, Pt, Hi)
  - Many voices available
  - **Streaming**: Supports real-time streaming playback (`--stream`)
  - **MLX Backend**: 30% faster on Apple Silicon (`--use-mlx`)
  - See [docs/KOKORO_GUIDE.md](docs/KOKORO_GUIDE.md) for usage details.
- **Best for**: High-quality offline synthesis without cloning


### 4. Marvis TTS (MLX) 🎭 Voice Cloning | ⚠️ Current Bug
- **Type**: Streaming TTS (250M params, MLX optimized)
- **Languages**: English, French, German
- **Features**:
  - ✨ **Voice cloning** support
  - Streaming generation
  - 4-bit quantization (500MB model size)
  - Optimized for Apple Silicon
  - Multilingual (EN/FR/DE)
  - See [docs/MARVIS_FIX.md](docs/MARVIS_FIX.md) for troubleshooting and details.
  - Speed and temperature control

### 5. Supertone (Supertonic) - Ultra-Fast TTS ⚡
- **Type**: ONNX-based TTS (66M params)
- **Features**:
  - Lightning fast (167× real-time on M4 Pro)
  - Ultra-lightweight on-device inference
  - Voice presets available
  - Configurable quality (inference steps)
  - **Streaming**: Supports pseudo-streaming playback (`--stream`)
  - See [docs/SUPERTONE_GUIDE.md](docs/SUPERTONE_GUIDE.md) for usage details.
- **Best for**: Speed-critical applications, on-device deployment

### 6. NeuTTS Air 🎭 Voice Cloning | 🍎 macOS Optimized
- **Type**: GGUF-based Voice Cloning TTS (0.5B params)
- **Languages**: English only
- **Features**:
  - ✨ **Instant voice cloning** with 3+ seconds of reference audio
  - On-device optimized (GGUF quantization)
  - NeuCodec for high-quality audio (24kHz)
  - See [docs/NEUTTS_AIR_GUIDE.md](docs/NEUTTS_AIR_GUIDE.md) for usage details.
- **Best for**: Voice cloning, on-device deployment

### 7. Dia2-1B - Streaming Dialogue TTS ⚠️ CUDA Only
- **Type**: Streaming multi-speaker TTS (1B params)
- **Features**:
  - Multi-speaker dialogue support (`[S1]`, `[S2]` tags)
  - Voice cloning via prefix audio
  - Streaming generation
  - Ultra-realistic conversational audio
  - See [docs/DIA2_GUIDE.md](docs/DIA2_GUIDE.md) for usage details.
- **⚠️ Critical Limitation**: 
  - **CUDA-only for practical use** (NVIDIA GPUs required)
  - **macOS Performance**: Extremely poor (CPU: 750x slower than real-time, MPS: unusable)
  - macOS users should use Kokoro, StyleTTS2, or Marvis instead
- **Best for**: Multi-speaker dialogue on Linux/Windows with NVIDIA GPU

### 8. Parakeet ASR 🌍 Multilingual | ⚠️ Dependency Conflict
- **Type**: Automatic Speech Recognition (0.6B params)
- **Languages**: 100+ languages (English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, and many more)
- **Backends**: 
  - **Mac (Apple Silicon)**: MLX optimized (fastest)
  - **Other**: NeMo toolkit (CUDA/CPU)
- **Features**: 
  - Multilingual support with automatic language detection
  - Timestamp generation (SRT)
- **Conflict**: Requires `transformers>=4.49.0` (incompatible with Chatterbox)

### 9. Canary ASR (Multilingual ASR/Translation)
- **Type**: ASR & Translation (1B params)
- **Features**: 
  - Supports 25 languages
  - Speech-to-Text (Transcription)
  - Speech-to-Speech Translation (via text output)
  - See [docs/CANARY_GUIDE.md](docs/CANARY_GUIDE.md) for usage details.
- **Backend**: NeMo toolkit

### 10. Granite ASR (IBM) ⚠️
- **Type**: Speech Recognition (3.3B params)
- **Model**: `ibm-granite/granite-speech-3.3-8b`
- **Features**: High accuracy, English optimized
- **Note**: Requires ~16GB RAM. May not work on systems with limited memory.


### 11. Whisper ASR 🌍 Multilingual
- **Type**: Encoder-Decoder ASR
- **Languages**: 99+ languages with translation to English
- **Features**: 
  - Multilingual transcription (auto-detect or manual)
  - Translation to English (`--target-language en`)
  - Timestamps support
  - Robust to accents and background noise
  - See [docs/WHISPER_GUIDE.md](docs/WHISPER_GUIDE.md) for usage details.
- **Model**: `openai/whisper-large-v3` (default) or `openai/whisper-tiny` (CPU)

### 12. HumAware-VAD
- **Type**: Voice Activity Detection
- **Features**: 
  - Distinguishes speech from humming
  - Adjustable threshold and timing parameters
  - See [docs/HUMAWARE_GUIDE.md](docs/HUMAWARE_GUIDE.md) for usage details.
- **Base**: Silero VAD (Fine-tuned)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd voice-cloning
   ```

2. **Install base dependencies:**

   ```bash
   uv sync
   ```

## Benchmarking

Run comprehensive benchmarks to measure performance on your hardware.

### Quick Start

```bash
uv run python benchmark.py
```

This will:
1. Generate test audio (using Kokoro TTS)
2. Benchmark all TTS models (Kitten, Kokoro, Marvis, Supertone)
3. Benchmark all ASR models (Parakeet, Canary, Whisper)
4. Benchmark VAD models (HumAware)
5. Save results with memory tracking to `docs/BENCHMARK_RESULTS.md`

### Advanced Options

```bash
# Test specific models only
python benchmark.py --models supertone,kitten

# Include streaming benchmarks (TTFA measurements)
python benchmark.py --include-streaming

# Include voice cloning tests
python benchmark.py --include-cloning

# Disable memory tracking
python benchmark.py --no-memory

# Skip specific tests
python benchmark.py --skip-asr  # Skip ASR benchmarks
python benchmark.py --skip-tts  # Skip TTS benchmarks

# Force specific device
python benchmark.py --device mps   # Apple Silicon
python benchmark.py --device cuda  # NVIDIA GPU
python benchmark.py --device cpu   # CPU only
```

### Documentation

- **[BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md)**: Complete results on MacBook Pro M3 8GB
- **[docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)**: Comprehensive benchmarking guide

### Sample Results (M3 MacBook Pro, 8GB)

| Model | Latency | RTF | Speed |
|-------|---------|-----|-------|
| **Supertone** | 583ms | 0.08 | 12× real-time |
| **KittenTTS** | 1.3s | 0.17 | 6× real-time |
| **Kokoro** | 3.7s | 0.52 | 2× real-time |
| **Parakeet (ASR)** | 2.6s | 0.37 | 2.7× real-time |

RTF < 1.0 means faster than real-time. See full results in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).



3. **Install model-specific dependencies:**

   **For Chatterbox:**

   ```bash
   pip install chatterbox-tts
   ```

   **For Parakeet ASR (Mac/Apple Silicon):**

   ```bash
   # Install the MLX CLI tool
   uv tool install parakeet-mlx
   ```

## Usage

### Command Line Interface

The main interface is through `main.py` with the following options:

```bash
uv run python main.py --model <model_name> --text "<text>" [options]
```

### Quick Start Examples

**1. Kitten TTS Nano (Fast & Lightweight)**
```bash
# Default (v0.2)
uv run python main.py --model kitten --text "Hello from Kitten TTS!" --output outputs/kitten.wav

# Specific Versions
uv run python main.py --model kitten-0.1 --text "Using version 0.1" --output outputs/kitten_v1.wav
uv run python main.py --model kitten-0.2 --text "Using version 0.2" --output outputs/kitten_v2.wav
```

**2. Kokoro (High Quality)**
```bash
# American English (Default)
uv run python main.py --model kokoro --text "Hello from Kokoro!" --output outputs/kokoro.wav --voice af_heart

# British English
uv run python main.py --model kokoro --text "Cheers mate!" --output outputs/kokoro_uk.wav --voice bf_emma --lang_code b
```

**3. Chatterbox (Voice Cloning)**
```bash
uv run python main.py --model chatterbox --text "Cloning this voice." --reference samples/ref.wav --output outputs/cloned.wav
```

**5. Parakeet ASR (Transcription)**
```bash
# Basic Transcription
uv run python main.py --model parakeet --reference samples/speech.wav --output outputs/transcript.txt

# With Timestamps (SRT)
uv run python main.py --model parakeet --reference samples/speech.wav --output outputs/subs.srt --timestamps
```

**6. Canary ASR (Multilingual)**
```bash
uv run python main.py --model canary --reference samples/speech.wav --output outputs/canary_transcript.txt
```

**7. Granite ASR (IBM)**
```bash
uv run python main.py --model granite --reference samples/speech.wav --output outputs/granite_transcript.txt
```

**8. Whisper ASR (OpenAI)**
```bash
uv run python main.py --model whisper --reference samples/speech.wav --output outputs/whisper_transcript.txt
```

**9. HumAware-VAD (Voice Detection)**
```bash
uv run python main.py --model humaware --reference samples/speech.wav --output outputs/vad_segments.txt
```

### Available Options

- `--model`: Choose from `chatterbox`, `kitten` (defaults to 0.2), `kitten-0.1`, `kitten-0.2`, `kokoro`, `parakeet`, `marvis`, `humaware`
- `--text`: Text to synthesize (required)
- `--reference`: Reference audio file for voice cloning (required for voice cloning models)
- `--output`: Output file path (required)
- `--output`: Output file path (required)
- `--speed`: Speech speed multiplier (default: 1.0)
- `--voice`: Voice for Kokoro model (default: af_heart)
- `--lang_code`: Language code for Kokoro (default: a)

### Programming Interface

You can also use the models directly in Python:


```python
from src.voice_cloning.kokoro import synthesize_speech

# Kokoro
result = synthesize_speech(
    text="Hello, world!",
    output_path="kokoro_output.wav",
    voice="af_heart"
)
```

## Project Structure

```
voice-cloning/
├── src/voice_cloning/          # Main package
│   ├── asr/                   # ASR models (Whisper, Parakeet, Canary, Granite)
│   ├── tts/                   # TTS models (Marvis, Kitten, Kokoro, Chatterbox)
│   └── vad/                   # Voice Activity Detection (HumAware)
├── main.py                    # CLI interface
├── scripts/                   # Utility and patch scripts
├── tests/                     # Test suite
├── docs/                      # Documentation
├── samples/                   # Sample audio files
├── outputs/                   # Generated audio/text (ignored)
├── models/                    # Downloaded models (ignored)
└── pyproject.toml            # Dependencies
```

## Troubleshooting

**Import Errors:**

- Ensure all dependencies are installed
- Use `uv run` to ensure proper virtual environment activation

**Audio File Issues:**

- Ensure reference audio files are in supported formats (WAV, MP3)
- Check that file paths are correct and accessible
