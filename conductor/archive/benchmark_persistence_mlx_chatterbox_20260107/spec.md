# Specification: Benchmark Reporting Improvements, MLX-Audio Update, and Chatterbox Turbo Integration

## Overview
This track aims to improve the persistence of benchmark results, synchronize with the latest `mlx-audio` improvements, and fully integrate `chatterbox` and `chatterbox-turbo` as distinct models with both PyTorch and MLX backend support.

## Functional Requirements

### 1. Persistent Benchmark Reporting
- **Smart Update Logic:** Modify the benchmarking suite to selectively update `docs/BENCHMARK_RESULTS.md` instead of overwriting it.
- **Master Comparison Table:** Maintain a central table for model performance comparison.
- **Strict Identification:** Use a unique identifier (Model Name + Backend + Hardware) to determine if a row should be updated or a new one added.

### 2. MLX-Audio Dependency Update
- **Audit & Optimize:** Conduct a comprehensive audit of the latest changes in the `mlx-audio` repository (https://github.com/Blaizzy/mlx-audio).
- **Model Amendments:** Apply relevant optimizations, bug fixes, and feature enhancements to all models utilizing `mlx-audio` (Kokoro, Marvis, Chatterbox).

### 3. Chatterbox & Chatterbox-Turbo Integration
- **Model Differentiation:** Treat `chatterbox` and `chatterbox-turbo` as two separate model entities.
- **Backend Support:** Ensure both models support PyTorch and MLX backends.
- **Feature Parity:** Implement all available features for both models (streaming, speed control, etc.).
- **Documentation:** 
    - Update `docs/CHATTERBOX_GUIDE.md`.
    - Create `docs/CHATTERBOX_TURBO_GUIDE.md`.
    - Ensure both are represented in `docs/BENCHMARK_RESULTS.md`.
- **Testing:** Expand the test suite to cover both models and their respective backends.

## Non-Functional Requirements
- **Performance:** Ensure the `mlx-audio` updates improve or maintain existing RTF and latency metrics.
- **Reliability:** The file modification logic for `BENCHMARK_RESULTS.md` must be robust against formatting errors.

## Acceptance Criteria
- [ ] `docs/BENCHMARK_RESULTS.md` correctly preserves historical data while updating relevant rows after a benchmark run.
- [ ] Latest `mlx-audio` optimizations are verified in the codebase.
- [ ] `chatterbox` and `chatterbox-turbo` are selectable and functional in the CLI/UI with both PT and MLX backends.
- [ ] Comprehensive tests pass for all Chatterbox variants.
- [ ] Documentation reflects the current state of both Chatterbox models.

## Out of Scope
- Adding new ASR or VAD models not mentioned in the request.
- Migrating non-MLX models to MLX unless they are part of the `mlx-audio` family.
