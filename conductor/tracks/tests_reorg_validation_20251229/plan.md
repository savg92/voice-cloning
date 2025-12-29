# Plan: Comprehensive Test Suite Reorganization and Full Model Validation

## Phase 1: Foundation and Reorganization [checkpoint: 10db901]
- [x] Task: Create structured directory hierarchy (`tests/tts/`, `tests/asr/`, `tests/vad/`, `tests/ui/`, `tests/core/`, `tests/data/`). c40f76a
- [x] Task: Move active test files into respective model folders and update paths/imports. 3b98d44
- [x] Task: Identify and delete obsolete/redundant test files from the root `tests/` directory. 3b98d44
- [x] Task: Conductor - User Manual Verification 'Foundation and Reorganization' (Protocol in workflow.md) 10db901

## Phase 2: Kokoro Comprehensive Validation
- [ ] Task: Refine `tests/tts/kokoro/test_kokoro_full.py` to cover all 9 languages and both backends (MLX/PyTorch).
- [ ] Task: Implement validation for Kokoro streaming and speed features.
- [ ] Task: Verify all Kokoro tests pass with robust output checks (file size > 0).
- [ ] Task: Conductor - User Manual Verification 'Kokoro Comprehensive Validation' (Protocol in workflow.md)

## Phase 3: Model-by-Model Feature Validation Plan
- [ ] Task: Create standardized test suites for **Kitten TTS** (testing multiple versions and voices).
- [ ] Task: Create standardized test suites for **Chatterbox/Marvis** (testing voice cloning and MLX specific features).
- [ ] Task: Create standardized test suites for **Supertone/NeuTTS Air** (testing specific ONNX/GGUF backends).
- [ ] Task: Create standardized test suites for **ASR Models** (Parakeet, Canary, Whisper) across multiple languages and formats.
- [ ] Task: Create standardized test suites for **HumAware VAD** (testing sensitivity thresholds).
- [ ] Task: Conductor - User Manual Verification 'Model-by-Model Feature Validation Plan' (Protocol in workflow.md)

## Phase 4: UI and Integration Finalization
- [ ] Task: Consolidate UI tests in `tests/ui/` ensuring integration with the new organized backend tests.
- [ ] Task: Perform a final end-to-end verification of the Web UI with all models.
- [ ] Task: Conductor - User Manual Verification 'UI and Integration Finalization' (Protocol in workflow.md)
