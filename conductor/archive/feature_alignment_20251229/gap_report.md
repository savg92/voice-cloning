# Feature Coverage Gap Report
Date: 2025-12-30
Track: Feature Alignment, Documentation Audit, and Benchmark Synchronization

## Overview
This report details the discrepancies found between the documentation (`docs/*_GUIDE.md`) and the test suite (`tests/`) for all supported models.

## TTS Models

### 1. Kokoro
*   **Status:** ✅ Well Covered
*   **Guide Features:** Streaming, MLX Backend, Multiple Voices, Speed Control, Multilingual.
*   **Test Coverage:** `test_kokoro_full.py` covers all languages, MLX/PyTorch backends, streaming, and speed control.
*   **Action:** None.

### 2. Kitten
*   **Status:** ✅ Well Covered
*   **Guide Features:** v0.1/v0.2 support, Voices, Speed Control, Streaming (pseudo).
*   **Test Coverage:** `test_kitten_full.py` covers versions, voices, speed, and streaming.
*   **Action:** None.

### 3. Supertone (Supertonic)
*   **Status:** ⚠️ Gaps Identified
*   **Guide Features:** Fast inference, Voices (F1/M1), Steps, CFG Scale, Streaming.
*   **Test Coverage:** `test_supertone_full.py` covers styles, streaming, and speed.
*   **Missing Tests:**
    *   `steps`: Parameter mentioned in guide but not tested.
    *   `cfg-scale`: Parameter mentioned in guide but not tested.
*   **Action:** Add tests for `steps` and `cfg_scale` parameters.

### 4. NeuTTS Air
*   **Status:** ⚠️ Gaps Identified
*   **Guide Features:** Voice Cloning, Auto-detect Ref Text, Backbone Selection (Q4/Q8).
*   **Test Coverage:** `test_neutts_air_full.py` covers cloning and auto-ref text.
*   **Missing Tests:**
    *   `backbone`: Parameter for selecting quantization (Q4 vs Q8) is not tested.
*   **Action:** Add test for `backbone` parameter.

### 5. Chatterbox
*   **Status:** ✅ Well Covered
*   **Guide Features:** MLX Backend (partial), Voice Cloning, Exaggeration, CFG Weight, Multilingual.
*   **Test Coverage:** `test_chatterbox_full.py` covers basic, cloning, multilingual, and controls (exaggeration/cfg).
*   **Action:** None.

### 6. Marvis
*   **Status:** ❌ Significant Gaps
*   **Guide Features:** Multilingual (En, Fr, De), Voice Cloning, Streaming, Speed, Temperature, Quantized flag.
*   **Test Coverage:** `test_marvis_full.py` covers basic (En), cloning, speed, and quantized flag.
*   **Missing Tests:**
    *   `streaming`: `--stream` feature is documented but not tested.
    *   `temperature`: `--temperature` parameter is documented but not tested.
    *   `multilingual`: Guide claims Fr/De support, but tests only check English.
*   **Action:** Add tests for streaming, temperature, and Fr/De languages.

### 7. CosyVoice
*   **Status:** ❌ Significant Gaps
*   **Guide Features:** Instruct Mode (Emotion/Style), Zero-Shot Cloning, MLX/PyTorch.
*   **Test Coverage:** `verify_cosyvoice.py` covers basic synthesis and zero-shot cloning.
*   **Missing Tests:**
    *   `instruct mode`: The `--emotion` / style control feature is a major selling point but is completely missing from verification.
*   **Action:** Add test case for `instruct` mode (emotion text).

### 8. Dia2
*   **Status:** ✅ Acceptable (given constraints)
*   **Guide Features:** Multi-speaker, Streaming, Voice Cloning.
*   **Test Coverage:** `test_dia2.py` mocks the library and verifies parameters (`cfg_scale`, `temperature`, `prefix_speaker`).
*   **Note:** Given the "Unusable on Mac" warning, mock-based testing is acceptable.
*   **Action:** None.

## ASR & VAD Models

### 9. Whisper
*   **Status:** ⚠️ Gaps Identified
*   **Guide Features:** Multilingual, Translation, Timestamps, MLX.
*   **Test Coverage:** `test_whisper_full.py` covers basic, MLX, multilingual, and translation.
*   **Missing Tests:**
    *   `timestamps`: Documented as a feature but not tested.
*   **Action:** Add test for `timestamps` parameter.

### 10. Parakeet
*   **Status:** ✅ Well Covered
*   **Guide Features:** Multilingual, Timestamps.
*   **Test Coverage:** `test_parakeet_full.py` covers basic and timestamps.
*   **Action:** None.

### 11. Canary
*   **Status:** ✅ Well Covered
*   **Guide Features:** Multilingual, Translation.
*   **Test Coverage:** `test_canary_full.py` covers basic and translation.
*   **Action:** None.

### 12. HumAware
*   **Status:** ✅ Well Covered
*   **Guide Features:** Thresholds, Durations.
*   **Test Coverage:** `test_humaware_full.py` covers parameters and thresholds.
*   **Action:** None.
