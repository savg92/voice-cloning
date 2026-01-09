# Specification: Chatterbox & Chatterbox-Turbo TTS Integration and UI Rework

## Overview
This track focuses on the integration of `ResembleAI/chatterbox-turbo` (and its MLX variant) alongside the existing standard Chatterbox implementation. A key objective is to rework the Gradio UI to treat them as two entirely separate, independent models with distinct controls and features.

## Functional Requirements

### 1. Distinct Model Implementations
- **Standard Chatterbox**: Verify and refine `src/voice_cloning/tts/chatterbox.py` (already implemented).
- **Chatterbox-Turbo**: Implement `src/voice_cloning/tts/chatterbox_turbo.py` as a standalone module.
- **Backend Support**: Ensure both support PyTorch and MLX backends with full feature support (cloning, multilingual, controls).

### 2. Gradio UI Rework
- **Separate Selectors**: List "Chatterbox" and "Chatterbox-Turbo" as unique options in the model dropdown.
- **Independent Parameter Groups**: Implement dedicated UI blocks for each model to manage their settings (Exaggeration, CFG Weight, etc.) in isolation.
- **Dynamic Presets**: Filter voice presets by language and hide preset selection when a reference audio is provided for cloning.

### 3. Benchmarking & Documentation
- **Turbo Benchmarking**: Implement `benchmarks/tts/chatterbox_turbo.py` and update the persistent results in `docs/BENCHMARK_RESULTS.md`.
- **Dedicated Documentation**: Update `docs/CHATTERBOX_GUIDE.md` and create `docs/CHATTERBOX_TURBO_GUIDE.md` to reflect the separate implementations.

### 4. Comprehensive Testing
- **Turbo Validation**: Create `tests/tts/chatterbox/test_chatterbox_turbo.py` covering synthesis, cloning, and multilingual features on both backends.
- **Regression Testing**: Ensure changes to the UI and the addition of Turbo do not break the existing Standard Chatterbox implementation.

## Non-Functional Requirements
- **Modularity**: Maintain strict separation between the standard and turbo codebases.
- **Performance**: Verify the speed advantages of the Turbo variant through benchmarking.

## Acceptance Criteria
- [ ] UI correctly distinguishes between Chatterbox and Chatterbox-Turbo with independent controls.
- [ ] Chatterbox-Turbo is fully functional via CLI and UI (PT and MLX).
- [ ] Voice presets are intelligently filtered/hidden based on language and cloning status.
- [ ] Benchmark results for both variants are persistent and compared.
- [ ] Documentation and tests are complete for both models.
