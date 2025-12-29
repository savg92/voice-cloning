# Specification: Fix Kokoro Multilingual Support

## 1. Overview
The user reported that Kokoro TTS fails to work properly for specific languages (German, Chinese, Japanese, Russian, Turkish) when using the Web UI. This track will investigate the root cause, likely related to language code mapping, model availability, or phonemizer requirements, and implement fixes.

## 2. Problem Description
- **Affected Languages:** German (`d`), Chinese (`z`), Japanese (`j`), Russian (`r`), Turkish (`t`).
- **Symptoms:** "Doesn't work properly" (could be silence, wrong language, error, or fallback to English).
- **Context:** Occurs in Web UI (`uv run main.py`).

## 3. Goals
- Verify the issue with a reproduction script.
- Identify the root cause (e.g., missing `misaki` for Japanese, missing `espeak` codes, or incorrect internal mapping).
- Ensure all supported Kokoro languages work correctly in both PyTorch and MLX backends (if applicable).

## 4. Implementation Plan
- Create reproduction script.
- Analyze `src/voice_cloning/tts/kokoro.py` and `utils.py`.
- Apply fixes.
- Verify with `tests/verify_all_features.py` or a new specific test.