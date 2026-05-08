# Supertonic 3 TTS Guide

Supertonic 3 is Supertone's latest ultra-fast, on-device multilingual TTS model. It supports 31 languages and expression tags for natural-sounding synthesis.

## Features

- **31 Languages**: Support for a wide range of global languages.
- **Expression Tags**: Add natural vocalizations like `<laugh>`, `<breath>`, and `<sigh>`.
- **Ultra-Fast**: Optimized for local inference using ONNX Runtime.
- **High Quality**: Improved reading stability and speaker similarity over v2.

## Supported Languages

`en`, `ko`, `ja`, `zh`, `es`, `fr`, `pt`, `de`, `it`, `ru`, `tr`, `vi`, `pl`, `nl`, `ar`, `hi`, `sv`, `da`, `fi`, `nb`, `cs`, `el`, `hu`, `ro`, `uk`, `id`, `ms`, `th`, `he`, `fa`, `ca`

## Usage

### CLI

Basic synthesis:
```bash
uv run main.py --model supertonic3 --text "Hello, how are you today?" --lang-code en
# Output saved to outputs/output.wav
```

Synthesis with expression tags:
```bash
uv run main.py --model supertonic3 --text "I am so happy to see you! <laugh>" --lang-code en
# Output saved to outputs/output.wav
```

Multilingual synthesis:
```bash
uv run main.py --model supertonic3 --text "Bonjour tout le monde" --lang-code fr
```

Changing voice:
```bash
uv run main.py --model supertonic3 --text "Testing different voices" --voice M1
```

### Options

- `--text`: The text to synthesize.
- `--lang-code`: Language code (default: `en`).
- `--voice`: Voice preset (e.g., `F1`, `M1`, `F2`, `M2`).
- `--speed`: Speech speed (default: `1.0`).
- `--output`: Output filename or full path (default: `output.wav`).
- `--output-dir`: Directory to save outputs (default: `outputs`). If `--output` is just a filename, it will be saved in this directory.

## Model Assets

The assets are automatically downloaded from Hugging Face on the first run to `models/supertonic3`.
Alternatively, you can manually clone the repository:

```bash
git lfs install
git clone https://huggingface.co/Supertone/supertonic-3 models/supertonic3
```

## Troubleshooting

- **Missing `onnxruntime`**: Ensure you have it installed: `uv add onnxruntime`.
- **Expression Tags**: Ensure tags are enclosed in angle brackets, e.g., `<laugh>`.
