# Plan: Finalize Proper Model Implementations and Validation

## Phase 1: Core Fixes and Verification
- [x] Task: Fix Kokoro MLX backend: add `--lang_code` and `--stream` flags.
- [x] Task: Update Kokoro default `lang_code` to 'a' and fix docstring.
- [x] Task: Implement CUDA check for Dia2 model in `tts_tab.py`.
- [x] Task: Consolidate `lang_map` in Kokoro and ensure consistency across backends. bf9aec0
- [x] Task: Verify Marvis and Chatterbox MLX backends for missing flags (e.g. `lang_code`). bf9aec0
- [x] Task: Fix `main.py` CLI to default to `web` model for easier UI launch.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md) [checkpoint: 17b9c96]

## Phase 2: Comprehensive Validation Suite
- [ ] Task: Create a comprehensive verification script `tests/verify_all_features.py`.
    - [ ] Subtask: Test Kokoro (PyTorch & MLX) with multiple languages.
    - [ ] Subtask: Test Voice Cloning (Chatterbox, Marvis, CosyVoice).
    - [ ] Subtask: Test Streaming feature for all supporting models.
    - [ ] Subtask: Test ASR models with different languages.
- [ ] Task: Run the validation suite and fix any discovered regressions.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)
