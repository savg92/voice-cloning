# Plan: Fix Kokoro Multilingual Support

## Phase 1: Investigation and Reproduction
- [x] Task: Create a reproduction script `tests/repro_kokoro_multilang.py` to confirm failure for de, zh, ja, ru, tr. b6fd2f2
- [x] Task: Analyze logs from reproduction to pinpoint error (e.g., "Language not supported", "Model not found").

## Phase 2: Implementation
- [x] Task: Fix language mapping or model loading in `src/voice_cloning/tts/kokoro.py` / `utils.py`. f717ec1
- [x] Task: Ensure necessary dependencies (e.g., `misaki` for JA/ZH) are installed/used if required by the backend. f717ec1

## Phase 3: Verification
- [x] Task: Run reproduction script to verify fixes.
- [x] Task: Manual verification in Web UI.