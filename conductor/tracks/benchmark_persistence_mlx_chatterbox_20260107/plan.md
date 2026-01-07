# Plan: Benchmark Reporting Improvements, MLX-Audio Update, and Chatterbox Turbo Integration

## Phase 1: MLX-Audio Update and Optimization
- [x] Task: Audit `mlx-audio` repository changes and identify optimization opportunities (0.2.10 supports Chatterbox, Turbo, and more)
- [~] Task: Update dependencies in `pyproject.toml` and lock file
- [ ] Task: Write failing performance/regression tests for Kokoro, Marvis, and Chatterbox (MLX backend)
- [ ] Task: Apply optimizations and bug fixes from `mlx-audio` to existing model implementations
- [ ] Task: Verify optimizations and ensure all tests pass for updated models
- [ ] Task: Conductor - User Manual Verification 'MLX-Audio Update and Optimization' (Protocol in workflow.md)

## Phase 2: Chatterbox & Chatterbox-Turbo Integration
- [ ] Task: Write failing unit tests for `chatterbox` and `chatterbox-turbo` (both variants, PT and MLX backends)
- [ ] Task: Refactor and implement `chatterbox` and `chatterbox-turbo` in `src/voice_cloning/tts/`
- [ ] Task: Implement backend selection logic (PyTorch vs MLX) for both Chatterbox variants
- [ ] Task: Ensure feature parity (streaming, speed control) for all Chatterbox combinations
- [ ] Task: Verify Chatterbox integration with existing CLI and UI components
- [ ] Task: Conductor - User Manual Verification 'Chatterbox & Chatterbox-Turbo Integration' (Protocol in workflow.md)

## Phase 3: Persistent Benchmark Reporting
- [ ] Task: Write failing tests for the `BENCHMARK_RESULTS.md` update logic (mocking file I/O)
- [ ] Task: Implement `BenchmarkResultsManager` to handle selective updates to the markdown file
- [ ] Task: Integrate `BenchmarkResultsManager` into the existing benchmark runner
- [ ] Task: Verify that running benchmarks preserves historical data and correctly updates matching rows
- [ ] Task: Conductor - User Manual Verification 'Persistent Benchmark Reporting' (Protocol in workflow.md)

## Phase 4: Documentation and Final Polish
- [ ] Task: Update `docs/CHATTERBOX_GUIDE.md` with new variant and backend details
- [ ] Task: Create `docs/CHATTERBOX_TURBO_GUIDE.md`
- [ ] Task: Perform a full benchmark run and verify the final state of `docs/BENCHMARK_RESULTS.md`
- [ ] Task: Final code review and ensure all Quality Gates are met
- [ ] Task: Conductor - User Manual Verification 'Documentation and Final Polish' (Protocol in workflow.md)
