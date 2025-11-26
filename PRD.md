# Product Requirements Document (PRD)

## Voice Cloning & ASR Research Toolkit

### 1. Executive Summary

The Voice Cloning & ASR Research Toolkit is a unified Python framework designed to evaluate, benchmark, and utilize state-of-the-art audio generation (TTS) and speech recognition (ASR) models. It provides a consistent CLI and Python API for interacting with diverse models ranging from lightweight CPU-optimized models (Kokoro) to large-scale foundation models (Canary-1B).

### 2. User Personas

- **AI Researcher:** Needs to quickly compare transcription accuracy or voice synthesis quality between different model architectures.
- **Developer:** Needs a reference implementation for integrating specific audio models into larger applications.
- **Content Creator:** Needs to generate voiceovers or transcribe audio content using local, privacy-focused AI.

### 3. Functional Requirements

#### 3.1 Text-to-Speech (TTS)

- **Input:** Text string, Reference Audio (for cloning), Language Code, Speed parameters.
- **Output:** Audio file (WAV).
- **Supported Models:**
  - **Kokoro:** High-quality, lightweight inference.
  - **OpenVoice (v1/v2):** Voice cloning with tone color conversion.
  - **Chatterbox:** Basic synthesis.
  - **Chatterbox:** Basic synthesis.
  - **Kitten:** Experimental lightweight TTS.
  - **Marvis:** Efficient, streaming-capable TTS (250M).
  - **Maya1:** 3B param expressive TTS with emotion control.

#### 3.2 Automatic Speech Recognition (ASR)

- **Input:** Audio file (WAV/FLAC), Source/Target Language.
- **Output:** Text transcript (TXT), Subtitles (SRT), or JSON metadata with timestamps.
- **Supported Models:**
  - **NVIDIA Canary-1B:** Multilingual ASR and Translation (AST).
  - **NVIDIA Parakeet:** Fast ASR (NeMo based).
  - **OpenAI Whisper:** Industry standard ASR.
  - **OpenAI Whisper:** Industry standard ASR.
  - **IBM Granite:** Time-series/Speech foundation model.
  - **Parakeet TDT:** Fast ASR (0.6B), with MLX support for Mac.

#### 3.3 Voice Activity Detection (VAD)

- **Input:** Audio file.
- **Output:** Timestamps of speech segments.
- **Supported Models:**
  - **HumAware-VAD:** Robust VAD that distinguishes speech from humming/singing.

#### 3.3 Interface

- **CLI:** A single entry point (`main.py`) capable of routing commands to specific models.
- **Standardization:** All ASR models must support a `transcribe_to_file` interface. All TTS models must support a `synthesize` interface.

### 4. Non-Functional Requirements

- **Modularity:** Adding a new model should not require refactoring existing logic.
- **Environment:** Dependencies managed via `uv` and `pyproject.toml`.
- **Performance:** Support for CUDA/MPS acceleration where available, with CPU fallback.
- **Cleanliness:** Generated artifacts must be isolated from source code.

### 5. Future Scope

- **Streaming:** Real-time audio processing.
- **Metrics:** Automated calculation of WER (Word Error Rate) for ASR and similarity scores for TTS.
- **Web UI:** A Gradio/Streamlit interface for visual testing.
