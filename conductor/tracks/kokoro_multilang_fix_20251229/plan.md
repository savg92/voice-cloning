# Plan: Fix Kokoro Multilingual Support

## Phase 1: Investigation and Reproduction
- [x] Task: Create a reproduction script `tests/repro_kokoro_multilang.py` to confirm failure for de, zh, ja, ru, tr. b6fd2f2
- [~] Task: Analyze logs from reproduction to pinpoint error (e.g., "Language not supported", "Model not found").

## Phase 2: Implementation
- [ ] Task: Fix language mapping or model loading in `src/voice_cloning/tts/kokoro.py` / `utils.py`.
- [ ] Task: Ensure necessary dependencies (e.g., `misaki` for JA/ZH) are installed/used if required by the backend.

## Phase 3: Verification
- [ ] Task: Run reproduction script to verify fixes.
- [ ] Task: Manual verification in Web UI.