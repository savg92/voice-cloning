# Plan: Feature Alignment, Documentation Audit, and Benchmark Synchronization

## Phase 1: Audit and Gap Identification [checkpoint: f1c3aa6]
- [x] Task: Audit `docs/*.md` and `tests/` to create a comprehensive "Feature Coverage Gap Report" for all models. [0a2efc6]
- [x] Task: Conductor - User Manual Verification 'Audit and Gap Identification' (Protocol in workflow.md) [f1c3aa6]

## Phase 2: TTS Model Alignment (Primary)
- [x] Task: Align **Kokoro & Kitten**: Fill gaps in tests (streaming, multilingual) or guides. [fbc02b2]
- [x] Task: Align **Supertone & NeuTTS Air**: Fill gaps in tests (styles, cloning) or guides. [0b7aabd]
- [x] Task: Align **Chatterbox & Marvis**: Fill gaps in tests (MLX features, cloning) or guides. [f5a593a]
- [x] Task: Align **CosyVoice & Dia2**: Fill gaps in tests (instruct, custom backends) or guides. [a5db0e7]
- [~] Task: Conductor - User Manual Verification 'TTS Model Alignment' (Protocol in workflow.md)

## Phase 3: Benchmark Synchronization
- [ ] Task: Extract Latency, RTF, and TTFA metrics from the latest successful outputs/logs.
- [ ] Task: Update `docs/BENCHMARK_RESULTS.md` with quantitative data and qualitative "Verification Notes".
- [ ] Task: Conductor - User Manual Verification 'Benchmark Synchronization' (Protocol in workflow.md)

## Phase 4: Secondary Scope and Final Review
- [ ] Task: Audit and align ASR (Whisper, Parakeet, Canary) and VAD (HumAware) models.
- [ ] Task: Perform a final project-wide documentation consistency check.
- [ ] Task: Conductor - User Manual Verification 'Secondary Scope and Final Review' (Protocol in workflow.md)
