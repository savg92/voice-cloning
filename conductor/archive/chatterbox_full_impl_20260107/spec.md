# Specification: Chatterbox TTS Full Implementation (Standard Variant)

## Overview
This track involves a complete "from scratch" implementation of the standard Chatterbox TTS model, including both the `ResembleAI/chatterbox` (PyTorch) and `mlx-community/chatterbox-4bit` (MLX) backends. The goal is to provide full feature parity, comprehensive documentation, benchmarking, and testing.

## Functional Requirements

### 1. Model Integration
- **PyTorch Backend:** Implement a wrapper for `ResembleAI/chatterbox` using the `chatterbox-tts` library.
- **MLX Backend:** Implement a wrapper for `mlx-community/chatterbox-4bit` using `mlx-audio>=0.2.10` for Apple Silicon optimization.
- **Feature Support:** 
    - Zero-shot voice cloning from audio references.
    - Multilingual synthesis support (23 languages).
    - Control parameters: Exaggeration (intensity) and CFG Weight (pacing).
    - Speed control.
    - Streaming synthesis support.

### 2. Documentation
- Create `docs/CHATTERBOX_GUIDE.md` containing:
    - Detailed installation instructions for both backends.
    - Usage examples for CLI and Python API.
    - Parameter reference guide.
    - Voice preset mapping for language consistency.

### 3. Benchmarking
- Create `benchmarks/tts/chatterbox.py` to measure:
    - Latency (Time to First Audio).
    - Real-Time Factor (RTF).
    - Memory usage (including MLX vs. PyTorch comparison on Mac).

### 4. Testing
- Create `tests/tts/chatterbox/test_chatterbox.py` with:
    - Basic synthesis tests for both backends.
    - Voice cloning verification.
    - Multilingual synthesis validation.
    - Parameter boundary testing (exaggeration, cfg).

### 5. CLI & UI Integration
- Update `main.py` to correctly route `--model chatterbox` requests to the new implementation.
- Integrate the model into the Gradio UI (`src/voice_cloning/ui/tts_tab.py`) with appropriate controls.

## Non-Functional Requirements
- **Performance:** MLX backend should leverage Apple Silicon's Neural Engine for sub-realtime synthesis.
- **Robustness:** Implement safe model loading (e.g., handling `torch.load` CUDA/CPU mismatches).

## Acceptance Criteria
- [ ] PyTorch and MLX backends are functional and selectable via CLI/UI.
- [ ] Voice cloning is verified working with reference audio.
- [ ] Multilingual synthesis is functional for supported languages.
- [ ] `docs/CHATTERBOX_GUIDE.md` is complete and accurate.
- [ ] Benchmark results are generated and persistent in `docs/BENCHMARK_RESULTS.md`.
- [ ] All unit tests in `tests/tts/chatterbox/` pass.

## Out of Scope
- Implementation of the "Turbo" variant (reserved for a separate track).
- Fine-tuning or training of the Chatterbox models.
