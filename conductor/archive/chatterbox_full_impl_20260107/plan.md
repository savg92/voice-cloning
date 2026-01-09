# Plan: Chatterbox TTS Full Implementation (Standard Variant)

## Phase 1: Foundation and Dependencies
- [x] Task: Update `pyproject.toml` with necessary Chatterbox dependencies and version relaxations
- [x] Task: Synchronize environment and install `chatterbox-tts`, `s3tokenizer`, and supporting libraries
- [x] Task: Write failing unit tests in `tests/tts/chatterbox/test_chatterbox.py` for basic synthesis and cloning
- [x] Task: Conductor - User Manual Verification 'Foundation and Dependencies' (Protocol in workflow.md)

## Phase 2: Core Backend Implementation
- [x] Task: Implement `src/voice_cloning/tts/chatterbox.py` with PyTorch backend support
- [x] Task: Implement MLX backend support in `src/voice_cloning/tts/chatterbox.py` using `mlx-audio`
- [x] Task: Implement safe `torch.load` monkeypatching to prevent CUDA/CPU errors and recursion
- [x] Task: Implement voice cloning logic and language-specific voice preset mapping
- [x] Task: Verify core synthesis and cloning features against unit tests
- [x] Task: Conductor - User Manual Verification 'Core Backend Implementation' (Protocol in workflow.md)

## Phase 3: CLI, UI, and Benchmarking Integration
- [x] Task: Update `main.py` CLI routing for the new Chatterbox implementation
- [x] Task: Implement `benchmarks/tts/chatterbox.py` for performance measurement
- [x] Task: Update Gradio UI in `src/voice_cloning/ui/tts_tab.py` with Chatterbox controls and filtering
- [x] Task: Verify UI interactions and CLI functionality
- [x] Task: Conductor - User Manual Verification 'CLI, UI, and Benchmarking Integration' (Protocol in workflow.md)

## Phase 4: Documentation and Final Polish
- [x] Task: Create comprehensive `docs/CHATTERBOX_GUIDE.md`
- [x] Task: Perform a full benchmark run and update `docs/BENCHMARK_RESULTS.md`
- [x] Task: Final code review and ensure all Quality Gates are met
- [x] Task: Conductor - User Manual Verification 'Documentation and Final Polish' (Protocol in workflow.md)