# Specification: Supertonic-2 Model Integration

## Overview
Integrate the `Supertone/supertonic-2` multilingual text-to-speech model into the `voice-cloning` platform. This model is an optimized, on-device TTS system using ONNX Runtime, supporting English, Korean, Spanish, Portuguese, and French. It will be implemented as a separate model from the existing `supertone` (v1) integration to maintain compatibility and allow for side-by-side comparison.

## Functional Requirements

### 1. Model Wrapper (`src/voice_cloning/tts/supertonic2.py`)
- Implement a `Supertonic2TTS` class based on the `Supertone/supertonic-2` architecture.
- **Weights Management:** Implement a hybrid approach that checks for models in `models/supertonic2` and automatically downloads them from Hugging Face if missing.
- **Multilingual Support:** Support language selection for EN, KO, ES, PT, and FR.
- **Controls:** Support adjustable inference steps (defaulting to recommended values) and speed control.
- **Audio Output:** Standard 24kHz or model-native sample rate output in WAV format.
- **Voice Presets:** If the model includes different voices/styles, include them in the implementation and UI.

### 2. Benchmarking (`benchmarks/tts/supertonic2.py`)
- Integrate into the existing benchmarking suite.
- Measure **Real-Time Factor (RTF)** and **Latency (TTFA)** across different languages and inference steps.

### 3. User Interface (Gradio)
- Add a new tab or section in the Gradio UI for Supertonic-2.
- Controls for:
    - Text Input
    - Language Selection (Dropdown)
    - Voice/Style Selection (Dropdown, if applicable)
    - Speed Slider
    - Inference Steps Slider
- Display performance metrics (Latency/RTF) after generation.

### 4. Documentation & Tests
- **Docs:** Create `docs/SUPERTONIC2_GUIDE.md` with setup instructions and usage examples.
- **Tests:** Implement unit tests in `tests/tts/supertonic2/` covering:
    - Model initialization and weight downloading.
    - Multilingual synthesis.
    - Speed and step parameter validation.

## Non-Functional Requirements
- **Performance:** Maintain the ultra-fast on-device performance characteristic of the model (aiming for < 0.1 RTF on Apple Silicon).
- **Privacy:** Ensure all inference remains local via ONNX Runtime.

## Out of Scope
- Integration with external cloud TTS APIs for this specific model.
- Fine-tuning or training scripts for Supertonic-2.
