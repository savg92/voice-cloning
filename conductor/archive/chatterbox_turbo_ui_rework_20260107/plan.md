# Plan: Chatterbox & Chatterbox-Turbo TTS Integration and UI Rework

## Phase 1: Chatterbox-Turbo Standalone Implementation
- [x] Task: Write failing unit tests for `src/voice_cloning/tts/chatterbox_turbo.py` (PT and MLX backends)
- [x] Task: Implement `src/voice_cloning/tts/chatterbox_turbo.py` with full feature parity (cloning, multilingual, controls)
- [x] Task: Verify Turbo implementation passes all tests and compare with Standard variant via CLI
- [x] Task: Conductor - User Manual Verification 'Chatterbox-Turbo Standalone Implementation' (Protocol in workflow.md)

## Phase 2: UI Rework and Model Separation
- [x] Task: Write failing structure tests for the Gradio UI rework
- [x] Task: Refactor `src/voice_cloning/ui/tts_tab.py` to list Standard and Turbo as separate model engines
- [x] Task: Implement dedicated, independent parameter groups for both variants in the UI
- [x] Task: Implement smart filtering for voice presets (language-specific and hide on cloning)
- [x] Task: Verify UI interactions and ensure no regressions for existing Standard variant
- [x] Task: Conductor - User Manual Verification 'UI Rework and Model Separation' (Protocol in workflow.md)

## Phase 3: Benchmarking and Comparison
- [x] Task: Implement `benchmarks/tts/chatterbox_turbo.py` for performance measurement
- [x] Task: Update `docs/BENCHMARK_RESULTS.md` with new persistent entries for both Standard and Turbo
- [x] Task: Verify that the benchmark results correctly highlight the Turbo variant's speed advantages
- [x] Task: Conductor - User Manual Verification 'Benchmarking and Comparison' (Protocol in workflow.md)

## Phase 4: Documentation and Final validation
- [x] Task: Update `docs/CHATTERBOX_GUIDE.md` to focus on the refined Standard implementation
- [x] Task: Create `docs/CHATTERBOX_TURBO_GUIDE.md` with specific optimization and usage details
- [x] Task: Final end-to-end verification of both models across all backends and interfaces
- [x] Task: Conductor - User Manual Verification 'Documentation and Final validation' (Protocol in workflow.md)