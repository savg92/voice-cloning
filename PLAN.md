# Project Plan

This file outlines the development plan for the voice-cloning project.

## Phase 1: Project Setup (Done)

- [x] Initialize Python project with `uv`
- [x] Create `pyproject.toml`
- [x] Create `README.md`
- [x] Create `.gitignore`
- [x] Create `src/voice_cloning` package structure
- [x] Create `INSTRUCTIONS.md`
- [x] Create `PLAN.md`

## Phase 2: Chatterbox Integration

- [ ] Add Chatterbox as a dependency.
- [ ] Create `src/voice_cloning/chatterbox.py`.
- [ ] Implement a function in `chatterbox.py` to synthesize speech from text and an audio sample.

## Phase 4: Command-Line Interface

- [x] Create `main.py`.
- [x] Implement a CLI to select the model (Chatterbox or OpenVoice).
- [x] The CLI will take an audio file path and text as input.
- [x] The CLI will save the output to a specified file.

## Phase 5: Documentation

- [x] Update `README.md` with setup and usage instructions.
- [x] Add docstrings to all functions.
- [x] Document OpenVoice v2 implementation.
- [x] Add comprehensive usage examples.

## Phase 6: Kokoro Integration

- [x] Add Kokoro as a dependency.
- [x] Create `src/voice_cloning/kokoro.py`.
- [x] Implement a function in `kokoro.py` to synthesize speech from text and an audio sample.
- [x] Test Kokoro implementation.

## Phase 7: Advanced Features & Testing

- [x] Implement OpenVoice v2 with multi-language support.
- [x] Add comprehensive CLI interface for all models.
- [x] Create detailed documentation and examples.
- [x] Create comprehensive test script for all models.
- [ ] Add automated testing suite.
- [ ] Performance benchmarking across models.
- [ ] Add audio quality evaluation metrics.

## Phase 8: Canary ASR Integration

- [x] Add Canary (nvidia/canary-1b-v2) wrapper for ASR
- [x] Integrate Canary into CLI (outputs transcript text file)
- [x] Implement full ASR and translation functionality for 25 languages
- [x] Add comprehensive Python API with timestamps and multiple output formats
- [x] Create test suite for Canary functionality
- [ ] Add tests comparing Canary transcription vs Whisper/Granite

## Project Status

✅ **COMPLETED**: Core functionality implemented and tested

- All 4 TTS models integrated (Chatterbox, OpenVoice v1, OpenVoice v2, Kokoro) + Canary ASR
- **Canary-1B-v2**: Full ASR and speech translation for 25 European languages
- Full CLI interface with comprehensive options
- Proper error handling and dependency checking
- Documentation and examples complete
- Test suite demonstrates working functionality

🔄 **NEXT STEPS**: Advanced features and optimization

- Automated testing and CI/CD
- Performance benchmarking
- Quality evaluation metrics
