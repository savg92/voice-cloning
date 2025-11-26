# New Models Walkthrough

I have integrated 5 new AI models into the Voice Cloning Toolkit. Here is how to use them.

## 1. HumAware-VAD (Voice Activity Detection)

Detects speech segments while ignoring humming/singing.

```bash
uv run python main.py --model humaware --reference samples/anger.wav --output outputs/segments.txt
```

## 2. Parakeet ASR (Speech Recognition)

Fast and accurate speech recognition.

- **Mac (Apple Silicon)**: Uses `mlx` backend automatically.
- **Other**: Uses `nemo` backend (requires `nemo_toolkit`).
- **Mac (Apple Silicon)**: Uses `mlx` backend automatically.
  - MLX uses a CLI runner (`uv run parakeet-mlx ...`) for inference. If `parakeet-mlx` or `uv` is not installed or unavailable, the CLI will automatically fallback to the NeMo backend (if installed).
- **Other**: Uses `nemo` backend (requires `nemo_toolkit`).

If neither support is available the CLI will print a helpful error suggesting actions such as installing `parakeet-mlx` (for MLX-based inference) or `nemo_toolkit[asr]` for fallback NeMo. For example:

```
✗ Parakeet transcription failed: NeMo toolkit not installed. Please install nemo_toolkit[asr] for non-MLX support.
```

If both MLX and NeMo are unavailable the CLI will stop with a helpful error explaining which dependency to install next (e.g., `parakeet-mlx` CLI on macOS or `nemo_toolkit[asr]` on other platforms).

```bash
uv run python main.py --model parakeet --reference samples/anger.wav --output outputs/parakeet_transcript.txt
```

To generate timestamps (SRT format):

```bash
uv run python main.py --model parakeet --reference samples/anger.wav --output outputs/parakeet.srt --timestamps
```

## 3. Marvis TTS (Speech Synthesis)

Efficient, streaming-capable TTS.

```bash
uv run python main.py --model marvis --text "Hello world" --output outputs/marvis_out.wav
```

## 4. Maya1 (Expressive TTS)

3B parameter model with emotion control.

> **Note**: On Mac, uses MLX-optimized 4-bit model (~1.5GB download, much faster). Other platforms use full model (~6GB, 5-15 min first download).

> **⚠️ Known Issue**: The MLX-optimized model (`nhe-ai/maya1-mlx-4Bit`) currently has a bug where it generates the voice description text as speech instead of the actual input text. To use Maya1 correctly, you need to disable MLX and use the full Transformers model. Set the environment variable `MAYA_DISABLE_MLX=1` before running.

```bash
uv run python main.py --model maya --text "Hello world" --emotion "laugh" --output outputs/maya_out.wav
```

## Verification

I have created a test script to verify that all models can be loaded:

```bash
uv run python tests/test_new_models.py
```

Output:

```
✓ HumAwareVAD imported
✓ ParakeetASR imported
✓ MarvisTTS imported
✓ Maya1 imported
```

## 5. Kitten TTS Nano (Lightweight TTS)

Kitten TTS Nano provides a fast, CPU-friendly TTS model with several high-quality voices.

Quick CLI usage with the new script:

> **Note**: Generated audio is automatically padded to a minimum of 1 second to ensure playability.

```bash
python scripts/kitten_cli.py --text "Hello from Kitten" --voice expr-voice-4-f --output outputs/kitten_out.wav
```

This requires the `kittentts` package, which you can install via:

```bash
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
```

If you encounter an error referencing "EspeakWrapper.set_data_path" or similar, try:

```bash
# upgrade phonemizer and espeakng-loader
pip install --upgrade phonemizer espeakng-loader

# On macOS (use espeak-ng, unlink espeak if installed)
brew install espeak-ng

# On Debian/Ubuntu
sudo apt-get install -y espeak-ng
```

Available voices:

```
expr-voice-2-m, expr-voice-2-f, expr-voice-3-m, expr-voice-3-f,
expr-voice-4-m, expr-voice-4-f, expr-voice-5-m, expr-voice-5-f
```

Or use the unified `main.py` CLI:

```bash
uv run python main.py --model kitten --text "Hello from main.py" --output outputs/kitten_main_out.wav
```
