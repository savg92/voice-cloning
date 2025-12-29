# Specification: Comprehensive Test Suite Reorganization and Full Model Validation

## 1. Overview
The current `tests/` directory is disorganized, which hinders development and verification. This track will reorganize the test suite into a category-and-model-based hierarchy and implement robust, feature-complete validation for all models. Starting with Kokoro, we will standardize testing for every supported TTS, ASR, and VAD model to ensure every unique capability (streaming, cloning, multilingual, etc.) is verified.

## 2. Functional Requirements
### 2.1 Standardized Directory Hierarchy
- Implement the structure: `tests/<category>/<model>/` (e.g., `tests/tts/kokoro/`, `tests/asr/whisper/`).
- Categories include: `tts`, `asr`, `vad`, `ui`, `core`, and `data`.
- Move all active tests into this structure and update internal paths/imports.

### 2.2 Robust Feature Validation (Per Model)
For **every** active model in the project, implement a comprehensive test suite that validates:
- **Core Functionality:** Basic synthesis/transcription/detection.
- **Unique Features:** Voice cloning (cloning models), Translation (Canary), Streaming (supported models), Speed/Pitch control.
- **Multilingual Support:** Validation across all supported language codes for that specific model.
- **Hardware Backends:** Standard PyTorch vs. optimized MLX (on Apple Silicon).
- **Edge Cases:** Empty inputs, very long/short inputs, invalid language codes.

### 2.3 Kokoro Full Validation (Initial Standard)
As the first model to be fully standardized:
- Implement `tests/tts/kokoro/test_kokoro_full.py`.
- Must pass for all 9 officially supported languages and both PyTorch/MLX backends.
- Must verify non-zero audio output and correct file generation.

### 2.4 Obsolete File Cleanup
- Identify and delete redundant, failing, or obsolete test scripts from the root `tests/` directory to maintain a clean workspace.

## 3. Acceptance Criteria
- [ ] All active tests are migrated to the `tests/<category>/<model>/` hierarchy.
- [ ] Comprehensive test suites exist for **all** models, covering their full feature set.
- [ ] `tests/tts/kokoro/test_kokoro_full.py` passes for all languages and backends.
- [ ] Root `tests/` directory contains only necessary configuration files (like `__init__.py`).
- [ ] All tests follow a robust pattern: verifying output existence, non-zero size, and handling device-specific logic (e.g., `pytest.mark.skipif`).
