# Specification: Finalize Proper Model Implementations and Validation

## 1. Overview
This track focuses on fixing specific issues reported by the user regarding model feature completeness and correctness, particularly for Kokoro MLX and models requiring CUDA.

## 2. Issues to Address
- **Kokoro MLX Streaming:** Enable and verify the `--stream` flag in the MLX backend.
- **Kokoro MLX Spanish Support:** Ensure `--lang_code` is passed correctly to the MLX backend to avoid defaulting to English pronunciation.
- **Dia2 CUDA Requirement:** Implement a proactive check for CUDA availability and show a user-friendly error message if not available.
- **Feature Parity:** Ensure all models (Marvis, Chatterbox, etc.) pass all relevant CLI flags to their respective backends (especially MLX).

## 3. Goals
- **Correctness:** Models must use the requested language and voice.
- **Robustness:** No crashes on unsupported hardware; helpful error messages instead.
- **Consistency:** All backends (PyTorch, MLX) should support the same feature set where technically possible.

## 4. Acceptance Criteria
- Kokoro MLX with `stream=True` plays audio correctly.
- Kokoro MLX with `lang_code='e'` (Spanish) uses correct Spanish pronunciation.
- Dia2 shows an error instead of crashing on macOS (MPS/CPU).
- Comprehensive verification script passes for all core models.
