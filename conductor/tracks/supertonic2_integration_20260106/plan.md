# Implementation Plan - Supertonic-2 Integration

## Phase 1: Core Implementation
- [x] Task: Create `src/voice_cloning/tts/supertonic2.py` 31382c4
    - [x] Sub-task: Implement `Supertonic2TTS` class structure.
    - [x] Sub-task: Implement model download logic (Hybrid approach: local check -> HF download).
    - [x] Sub-task: Implement text pre-processing and ONNX inference pipeline.
    - [x] Sub-task: Implement multilingual support (EN, KO, ES, PT, FR).
    - [x] Sub-task: Add support for `speed` and `steps` parameters.
- [x] Task: Unit Tests 31382c4
    - [x] Sub-task: Create `tests/tts/supertonic2/` directory.
    - [x] Sub-task: Implement `test_init.py` (loading/downloading).
    - [x] Sub-task: Implement `test_synthesis.py` (multilingual generation).
    - [x] Sub-task: Run tests to verify core functionality.
- [ ] Task: Conductor - User Manual Verification 'Core Implementation' (Protocol in workflow.md)

## Phase 2: Benchmarking Integration
- [ ] Task: Create `benchmarks/tts/supertonic2.py`
    - [ ] Sub-task: Implement `Supertonic2Benchmark` class inheriting from `TTSBenchmark`.
    - [ ] Sub-task: Register benchmark in `benchmarks/config.py` (if applicable) or runner.
- [ ] Task: Run Benchmarks
    - [ ] Sub-task: Execute benchmark on Apple Silicon (if available) or current env.
    - [ ] Sub-task: Verify RTF and Latency metrics are captured correctly.
- [ ] Task: Conductor - User Manual Verification 'Benchmarking Integration' (Protocol in workflow.md)

## Phase 3: UI & Documentation
- [ ] Task: UI Integration
    - [ ] Sub-task: Update Gradio interface (likely in `src/voice_cloning/ui/` or `main.py`) to include Supertonic-2.
    - [ ] Sub-task: Add Language Dropdown, Speed Slider, and Steps control.
    - [ ] Sub-task: Add "Voice" dropdown if multiple speaker files are found in the model config (conditional feature).
- [ ] Task: Documentation
    - [ ] Sub-task: Create `docs/SUPERTONIC2_GUIDE.md`.
    - [ ] Sub-task: Update `README.md` to list Supertonic-2 support.
- [ ] Task: Conductor - User Manual Verification 'UI & Documentation' (Protocol in workflow.md)
