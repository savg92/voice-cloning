# Specification: Feature Alignment, Documentation Audit, and Benchmark Synchronization

## 1. Overview
This track aims to ensure absolute consistency between the project's technical documentation (guides) and the actual implemented/tested features. We will audit the `docs/` folder against the recently reorganized `tests/` suite, filling gaps in either the tests or the guides. Additionally, we will update `docs/BENCHMARK_RESULTS.md` with high-fidelity performance data and qualitative findings from recent runs.

## 2. Functional Requirements
### 2.1 Documentation & Test Audit (Primary: TTS)
- **Primary Scope:** Audit all TTS models: Kokoro, Kitten, Chatterbox, Marvis, Supertone, NeuTTS Air, CosyVoice, and Dia2.
- **Alignment Check:** Ensure every feature described in `docs/<MODEL>_GUIDE.md` has a corresponding validation in `tests/tts/<model>/`.
- **Gap Resolution:** 
    - If a feature is documented but not tested: Implement the test.
    - If a feature is tested but not documented: Update the guide.
- **Secondary Scope:** Perform the same audit for ASR (Whisper, Parakeet, Canary) and VAD (HumAware) models as time permits.

### 2.2 Benchmark Results Synchronization
- **Metric Extraction:** Gather Latency, RTF, and TTFA data from the most recent successful test/benchmark executions.
- **Hardware Comparison:** Document performance deltas between PyTorch and MLX backends on Apple Silicon where applicable.
- **Qualitative Findings:** Add "Verification Notes" for each model summarizing which features (Streaming, Multilingual, Cloning, etc.) are confirmed working.
- **Persistence:** Update `docs/BENCHMARK_RESULTS.md` while preserving the existing structure and historical context.

## 3. Acceptance Criteria
- [ ] Every active TTS model has 100% alignment between its `docs/` guide and `tests/` suite.
- [ ] All features (e.g., streaming, speed control, voice cloning) are explicitly verified in tests and documented in guides.
- [ ] `docs/BENCHMARK_RESULTS.md` contains a dedicated section for recent findings, including a mix of quantitative metrics and qualitative feature status.
- [ ] No regression in test pass rates for the recently reorganized test suite.

## 4. Out of Scope
- Implementation of entirely new model architectures (only feature completion/alignment for existing ones).
- Redesigning the Gradio UI (documentation alignment only).
