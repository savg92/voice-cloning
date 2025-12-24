# Plan: Gradio Web Interface

## Phase 1: Setup and Basic TTS Interface
- [x] Task: Create `src/voice_cloning/ui/` directory and `__init__.py`. 47e0846
- [x] Task: Implement basic Gradio scaffolding in `src/voice_cloning/ui/app.py`. ef304b8
- [x] Task: Create `src/voice_cloning/ui/tts_tab.py` for TTS logic. 1437c78
- [x] Task: Implement integration for **Kokoro** and **Kitten** TTS in the UI. fc4b6d5
    - [ ] Subtask: Write integration tests for UI-to-Model calls.
    - [ ] Subtask: Implement the UI components and callback logic.
- [x] Task: Add CLI command `--web` or `web` to `main.py` to launch the interface. 75fe8ad
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Advanced TTS and Voice Cloning
- [ ] Task: Implement integration for **Chatterbox** and **Marvis** (Voice Cloning) in `src/voice_cloning/ui/tts_tab.py`.
    - [ ] Subtask: Add file upload component for reference audio.
    - [ ] Subtask: Update logic to handle reference audio input.
- [ ] Task: Implement integration for **CosyVoice2** (if available/stable).
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: ASR and VAD Interfaces
- [ ] Task: Create `src/voice_cloning/ui/asr_tab.py` for ASR logic.
- [ ] Task: Implement integration for **Whisper** and **Parakeet** ASR.
    - [ ] Subtask: Write integration tests.
    - [ ] Subtask: Implement file upload and transcription display.
- [ ] Task: Create `src/voice_cloning/ui/vad_tab.py` for VAD logic.
- [ ] Task: Implement integration for **HumAware** VAD.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Polish and Documentation
- [ ] Task: Refine UI layout (grouping inputs, adding labels/markdown instructions).
- [ ] Task: Add error handling and user feedback (e.g., "Model loading...").
- [ ] Task: Update `README.md` and create `docs/WEB_UI_GUIDE.md`.
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
