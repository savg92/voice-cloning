# Canary ASR - Usage Guide

**NVIDIA Canary-1B-v2** is a powerful multilingual model that supports:
- **Automatic Speech Recognition (ASR)**: Transcribing speech to text
- **Automatic Speech Translation (AST)**: Translating speech from one language to another

## Features

- **Multilingual**: Supports 25 languages
- **Translation**: Can translate any supported language to English, or English to any supported language
- **Punctuation & Capitalization**: Generates fully formatted text
- **Timestamps**: Not natively supported in this wrapper yet (coming soon)

## Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `de` | German | `fr` | French |
| `es` | Spanish | `it` | Italian | `pt` | Portuguese |
| `pl` | Polish | `ru` | Russian | `nl` | Dutch |
| `cs` | Czech | `ro` | Romanian | `hu` | Hungarian |
| `bg` | Bulgarian | `hr` | Croatian | `lt` | Lithuanian |
| `lv` | Latvian | `sk` | Slovak | `sl` | Slovenian |
| `sv` | Swedish | `da` | Danish | `fi` | Finnish |
| `no` | Norwegian | `el` | Greek | `tr` | Turkish |
| `uk` | Ukrainian | | | | |

## Usage Examples

### 1. Basic Transcription (English)
```bash
uv run python main.py --model canary \
  --reference samples/speech.wav \
  --output outputs/transcript.txt
```

### 2. Multilingual Transcription (e.g., French)
Specify the source language with `--language`:
```bash
uv run python main.py --model canary \
  --reference samples/french_speech.wav \
  --language fr \
  --output outputs/french_transcript.txt
```

### 3. Speech Translation (French -> English)
Specify source language and target language:
```bash
uv run python main.py --model canary \
  --reference samples/french_speech.wav \
  --language fr \
  --target-language en \
  --output outputs/translation_en.txt
```

### 4. Speech Translation (English -> German)
```bash
uv run python main.py --model canary \
  --reference samples/speech.wav \
  --language en \
  --target-language de \
  --output outputs/translation_de.txt
```

## Troubleshooting

- **Memory Usage**: Canary is a 1B parameter model. It requires ~4GB VRAM/RAM.
- **First Run**: The first run will download the model (~2GB), which may take some time.
- **Device**: Automatically uses CUDA (NVIDIA GPU) or MPS (Mac Apple Silicon) if available.
