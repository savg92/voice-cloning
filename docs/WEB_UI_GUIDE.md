# Web Interface Guide

The Voice Cloning & ASR Research Toolkit includes a user-friendly Gradio web interface for interactive testing and comparison of models.

## Launching the Web UI

To start the web interface, run the following command from the project root:

```bash
uv run python main.py --model web
```

By default, the interface will be available at `http://127.0.0.1:7860`.

## Features

### 🎭 Text-to-Speech (TTS)
- **Model Selection**: Switch between Kokoro, Kitten, Chatterbox, Marvis, and CosyVoice.
- **Voice Cloning**: Upload a reference audio file for zero-shot cloning (supported by Chatterbox, Marvis, and CosyVoice).
- **Advanced Parameters**:
    - **Speed**: Adjust the playback speed multiplier.
    - **MLX Acceleration**: Toggle MLX optimization for Apple Silicon.
    - **Model-Specific Settings**: Configure voice styles, CFG weights, sampling temperature, etc.

### 📝 Speech Recognition (ASR)
- **Model Selection**: Choose between Whisper, Parakeet, and Canary.
- **Translation**: Whisper and Canary support speech-to-English translation.
- **Timestamps**: Enable word or sentence-level timestamps (SRT format for Parakeet).
- **MLX Support**: Fast inference using MLX for Whisper.

### 🔍 Voice Activity Detection (VAD)
- **HumAware-VAD**: Identify speech segments in any audio file.
- **Adjustable Thresholds**: Fine-tune the detection sensitivity and minimum segment duration.
- **JSON Output**: Export detected segments as JSON metadata.

## Troubleshooting

- **Memory Issues**: Loading multiple large models (e.g. Canary and Whisper) simultaneously may exceed your RAM/VRAM. Restart the server to clear the cache if needed.
- **Port Conflicts**: If port `7860` is in use, Gradio will automatically try the next available port.
