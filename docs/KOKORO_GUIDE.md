# Kokoro TTS - Usage Guide

**Kokoro** is a high-quality, lightweight neural TTS model (82M parameters) that supports multiple languages and voices.

## Features

- **High Quality**: Near-human naturalness.
- **Multilingual**: Supports English (US/UK), French, Japanese, Chinese, Spanish, Italian, Portuguese, Hindi.
- **Multiple Voices**: Dozens of voices across different languages.
- **Speed Control**: Adjust speaking rate.

## Usage Examples

### 1. Basic Synthesis (Default: US English, af_heart)
```bash
uv run python main.py --model kokoro \
  --text "Hello, this is Kokoro TTS." \
  --output outputs/kokoro.wav
```

### 2. Change Voice
```bash
uv run python main.py --model kokoro \
  --text "This is the Nicole voice." \
  --voice af_nicole \
  --output outputs/nicole.wav
```

### 3. Change Language
Use `--lang_code` to select the language.
```bash
# British English
uv run python main.py --model kokoro \
  --text "Hello from London." \
  --lang_code b \
  --voice bf_emma \
  --output outputs/british.wav

# Japanese
uv run python main.py --model kokoro \
  --text "こんにちは" \
  --lang_code j \
  --voice jf_alpha \
  --output outputs/japanese.wav
```

### 4. Adjust Speed
```bash
uv run python main.py --model kokoro \
  --text "Speaking fast." \
  --speed 1.2 \
  --output outputs/fast.wav
```

### 5. Streaming Playback
Enable streaming to play audio chunks as they are generated.
```bash
uv run python main.py --model kokoro \
  --text "This is sentence one. This is sentence two. This is sentence three." \
  --stream \
  --output outputs/kokoro_stream.wav
```

## Supported Languages & Codes

| Code | Language | Code | Language |
|------|----------|------|----------|
| `a` | American English | `b` | British English |
| `f` | French | `j` | Japanese |
| `z` | Mandarin Chinese | `e` | Spanish |
| `i` | Italian | `p` | Brazilian Portuguese |
| `h` | Hindi | | |

## Available Voices

### 🇺🇸 American English ('a')
**Female**: `af_heart` (Default), `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`
**Male**: `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

### 🇬🇧 British English ('b')
**Female**: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`
**Male**: `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

### 🇫🇷 French ('f')
`ff_siwis`

### 🇯🇵 Japanese ('j')
**Female**: `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`
**Male**: `jm_kumo`

### 🇨🇳 Mandarin Chinese ('z')
**Female**: `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`
**Male**: `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

### 🇪🇸 Spanish ('e')
`ef_dora`, `em_alex`, `em_santa`

### 🇮🇹 Italian ('i')
`if_sara`, `im_nicola`

### 🇧🇷 Brazilian Portuguese ('p')
`pf_dora`, `pm_alex`, `pm_santa`

### 🇮🇳 Hindi ('h')
`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`
