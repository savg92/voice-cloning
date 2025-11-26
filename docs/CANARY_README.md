# NVIDIA Canary-1B-v2 Implementation

This directory contains the implementation of the NVIDIA Canary-1B-v2 model for automatic speech recognition (ASR) and speech translation (AST).

## Overview

Canary-1B-v2 is a powerful 1-billion parameter model built for high-quality speech transcription and translation across 25 European languages. It features:

- **Speech Transcription (ASR)** for 25 languages
- **Speech Translation (AST)** from English → 24 languages
- **Speech Translation (AST)** from 24 languages → English
- Automatic punctuation and capitalization
- Word-level and segment-level timestamps
- Noise robustness and long-form inference

## Supported Languages

The model supports 25 European languages:

| Code | Language  | Code | Language | Code | Language   |
| ---- | --------- | ---- | -------- | ---- | ---------- |
| bg   | Bulgarian | hr   | Croatian | cs   | Czech      |
| da   | Danish    | nl   | Dutch    | en   | English    |
| et   | Estonian  | fi   | Finnish  | fr   | French     |
| de   | German    | el   | Greek    | hu   | Hungarian  |
| it   | Italian   | lv   | Latvian  | lt   | Lithuanian |
| mt   | Maltese   | pl   | Polish   | pt   | Portuguese |
| ro   | Romanian  | sk   | Slovak   | sl   | Slovenian  |
| es   | Spanish   | sv   | Swedish  | ru   | Russian    |
| uk   | Ukrainian |      |          |      |            |

## Files

- `canary.py` - Simple interface compatible with existing CLI
- `canary-1b-v2.py` - Full-featured implementation with advanced options
- `test_canary.py` - Comprehensive test suite

## Usage

### Basic ASR (Speech-to-Text)

```bash
# Using main CLI
python main.py --model canary --reference audio.wav --output transcript.txt

# Using canary-1b-v2.py directly
python src/voice_cloning/canary-1b-v2.py audio.wav --source-lang en --target-lang en
```

### Speech Translation

```bash
# English to French
python src/voice_cloning/canary-1b-v2.py audio.wav --source-lang en --target-lang fr

# German to English
python src/voice_cloning/canary-1b-v2.py audio.wav --source-lang de --target-lang en
```

### Advanced Features

```bash
# With timestamps
python src/voice_cloning/canary-1b-v2.py audio.wav --timestamps

# JSON output format
python src/voice_cloning/canary-1b-v2.py audio.wav --format json

# SRT subtitle format (requires timestamps)
python src/voice_cloning/canary-1b-v2.py audio.wav --timestamps --format srt

# List supported languages
python src/voice_cloning/canary-1b-v2.py --list-languages
```

## Python API

### Simple Interface

```python
from src.voice_cloning.canary import transcribe_to_file, get_canary

# Basic transcription
transcript_path = transcribe_to_file(
    audio_path="audio.wav",
    output_path="transcript.txt",
    source_lang="en",
    target_lang="en"
)

# Get model instance
canary = get_canary()
result = canary.transcribe("audio.wav", source_lang="en", target_lang="fr")
print(result['text'])
```

### Advanced Interface

```python
from src.voice_cloning.canary_1b_v2 import CanaryV2Model

# Create model instance
model = CanaryV2Model()

# Single file transcription
result = model.transcribe_single(
    audio_path="audio.wav",
    source_lang="en",
    target_lang="fr",
    timestamps=True
)

# Batch processing
results = model.transcribe(
    audio_files=["audio1.wav", "audio2.wav"],
    source_lang="en",
    target_lang="de",
    batch_size=2
)

# Save results in different formats
model.save_transcription(result, "output.txt", format="txt")
model.save_transcription(result, "output.json", format="json")
model.save_transcription(result, "output.srt", format="srt")
```

## Requirements

The implementation requires:

- Python 3.10-3.12
- NeMo Toolkit 2.4.0+
- PyTorch
- Additional dependencies (automatically installed via uv)

Dependencies are automatically managed through the project's `pyproject.toml`.

## Performance

- **Model Size**: 978M parameters
- **Memory**: Requires at least 6GB GPU memory
- **Speed**: Processes audio in approximately 20-25 seconds per file on modern GPUs
- **Accuracy**: State-of-the-art performance among models of similar size

## Model Architecture

Canary-1B-v2 uses an encoder-decoder architecture:

- **Encoder**: FastConformer with 32 layers
- **Decoder**: Transformer with 8 layers
- **Tokenizer**: SentencePiece with 16,384 vocabulary
- **Total Parameters**: ~978 million

## License

The model is released under the CC BY 4.0 license, making it suitable for both commercial and non-commercial use.

## Testing

Run the test suite to verify functionality:

```bash
python test_canary.py
```

The test suite covers:

- Language support validation
- ASR transcription
- Speech translation (EN→FR, EN→DE)
- CLI integration

## Using uv (dependency & runtime helper)

This project uses `uv` to manage a lightweight, reproducible environment and to install dependencies. Below are common `uv` commands and examples you can run from the project root (`/Users/savg/Desktop/voice-cloning`). Replace `python` with the appropriate interpreter if your environment differs.

1. Sync project dependencies (reads `pyproject.toml` and installs packages):

```bash
# from project root
uv sync
```

2. Add a package quickly (for missing dependency or experimental installs):

```bash
# add a package to the active environment
uv add <package-name>
# example:
uv add pyannote.audio pyannote.core
```

3. Run Python commands inside the `uv` environment (useful for quick tests):

```bash
uv run python -c "from src.voice_cloning.canary import get_canary; print('ok')"
```

4. Run the CLI or test suite inside the environment:

```bash
# Run the project CLI
uv run python main.py --model canary --reference sample_voices/anger.wav --output test_canary_output

# Run the test suite
uv run python test_canary.py
```

Notes:

- The Canary model downloads a ~6.3GB NeMo checkpoint on first load; make sure you have sufficient disk space and a stable network connection.
- If `uv sync` fails due to version conflicts, consider adjusting `pyproject.toml` (we pinned compatible NeMo/transformers versions) or use `uv add` to install small missing packages as needed.
- Use `uv add` instead of using `pip` directly inside the `uv` environment to keep the project manifest and environment in sync.

## Examples

### Example Outputs

**Input Audio**: "Can you prove it? What proof do you have? Can you prove it?"

**ASR (EN→EN)**: "Can you prove it? What proof do you have? Can you prove it?"

**Translation (EN→FR)**: "Pouvez-vous le prouver? Quelles preuves avez-vous? Pouvez-vous le prouver?"

**Translation (EN→DE)**: "Können Sie es beweisen? Welche Beweise haben Sie? Können Sie es beweisen?"

## References

- [Hugging Face Model Page](https://huggingface.co/nvidia/canary-1b-v2)
- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html)
- [NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)
