# Plan: Benchmark Reporting Improvements, MLX-Audio Update, and Chatterbox Turbo Integration

## Phase 1: MLX-Audio Update and Optimization
- [x] Task: Audit `mlx-audio` repository changes and identify optimization opportunities (0.2.10 supports Chatterbox, Turbo, and more)
- [x] Task: Update dependencies in `pyproject.toml` and lock file (0.2.10)
- [x] Task: Write failing performance/regression tests for Kokoro, Marvis, and Chatterbox (MLX backend) (Verified working)
- [x] Task: Apply optimizations and bug fixes from `mlx-audio` to existing model implementations (Identified `strict=False` in `load_model`)
- [x] Task: Verify optimizations and ensure all tests pass for updated models (Verification script passed)
- [x] Task: Conductor - User Manual Verification 'MLX-Audio Update and Optimization' (Protocol in workflow.md)

## Phase 2: Chatterbox & Chatterbox-Turbo Integration
- [x] Task: Write failing unit tests for `chatterbox` and `chatterbox-turbo` (both variants, PT and MLX backends)
- [x] Task: Refactor and implement `chatterbox-turbo` in `src/voice_cloning/tts/`
- [x] Task: Implement backend selection logic (PyTorch vs MLX) for both Chatterbox variants
- [x] Task: Ensure feature parity (streaming, speed control) for all Chatterbox combinations
- [x] Task: Verify Chatterbox integration with existing CLI and UI components (Verified with `test_variants.py`)
- [x] Task: Conductor - User Manual Verification 'Chatterbox & Chatterbox-Turbo Integration' (Protocol in workflow.md)

## Phase 3: Persistent Benchmark Reporting
- [x] Task: Write failing tests for the `BENCHMARK_RESULTS.md` update logic (mocking file I/O)
- [x] Task: Implement `BenchmarkResultsManager` to handle selective updates to the markdown file
- [x] Task: Integrate `BenchmarkResultsManager` into the existing benchmark runner
- [x] Task: Verify that running benchmarks preserves historical data and correctly updates matching rows (Verified with `test_benchmark_results_manager.py`)
- [x] Task: Conductor - User Manual Verification 'Persistent Benchmark Reporting' (Protocol in workflow.md)

## Phase 4: Documentation and Final Polish
- [x] Task: Update `docs/CHATTERBOX_GUIDE.md` with new variant and backend details
- [x] Task: Create `docs/CHATTERBOX_TURBO_GUIDE.md`
- [x] Task: Perform a full benchmark run and verify the final state of `docs/BENCHMARK_RESULTS.md`
- [x] Task: Final code review and ensure all Quality Gates are met
- [x] Task: Conductor - User Manual Verification 'Documentation and Final Polish' (Protocol in workflow.md)
