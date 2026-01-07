# Tech Stack

## Programming Language
*   **Python:** 3.12 (Supported range: 3.10 to 3.13)

## Package Management
*   **uv:** Fast Python package manager used for dependency management and environment isolation.

## Core AI & Machine Learning
*   **Apple Silicon Optimization (MLX):** `mlx`, `mlx-audio`, `mlx-lm` are used for high-performance inference on M-series chips. `mlx-audio` is specifically used for optimized TTS (Kokoro, Marvis, Chatterbox).
*   **Deep Learning Frameworks:** `torch` (PyTorch), `torchaudio`, and `pytorch-lightning`.
*   **Model Integration:** `transformers` (Hugging Face), `huggingface-hub`, `safetensors`.
*   **Engines:**
    *   **ASR:** `nemo-toolkit`, `faster-whisper`, `whisper-timestamped`, `mlx-whisper`.
    *   **TTS:** `kokoro`, `kittentts`, `onnxruntime` (for Supertone & Supertonic-2), `chatterbox-tts`, `cosyvoice`, `mlx-audio-plus` (for CosyVoice2), `llama-cpp-python` (for NeuTTS Air), `dia2`.
    *   **VAD:** `pyannote-audio`.

## Audio Processing & Utilities
*   **Audio Manipulation:** `librosa`, `pydub`, `soundfile`, `scipy`, `pyloudnorm`.
*   **Phonetics:** `phonemizer`, `misaki`, `unidic`.
*   **CLI & Interface:** `typer`, `gradio`.

## Infrastructure & Architecture
*   **Modular Design:** The project is structured as a package (`src/voice_cloning`) with distinct modules for different audio tasks.
*   **Build System:** `hatchling` is used for building wheels and source distributions.
