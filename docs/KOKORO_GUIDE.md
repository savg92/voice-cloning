# Kokoro TTS - Usage Guide

**Kokoro** is a high-quality, lightweight neural TTS model (82M parameters) that supports multiple languages and voices.

## Installation

Install Kokoro dependencies:

```bash
pip install kokoro
```

### MLX Backend (Apple Silicon)

For optimized performance on Apple Silicon (M1/M2/M3):

```bash
pip install mlx-audio
```

## Features

- **8 Languages**: American English, British English, Spanish, French, Italian, Portuguese, Japanese, Chinese
- **Streaming Support**: Real-time audio generation
- **MLX Backend**: Up to 30% faster on Apple Silicon
- **Multiple Voices**: Various voice styles available
- **Speed Control**: Adjust speaking rate.

## Usage

### Basic Synthesis (PyTorch)

```bash
uv run python main.py --model kokoro \
    --text "Hello, this is Kokoro TTS" \
    --output output.wav \
    --voice af_heart \
    --lang-code e
```

### With MLX Backend (Apple Silicon)

```bash
uv run python main.py --model kokoro \
    --text "Hello, this is Kokoro with MLX" \
    --output output.wav \
    --use-mlx \
    --voice af_heart
```

### With Streaming

```bash
uv run python main.py --model kokoro \
    --text "This will stream as it generates" \
    --output outputs/kokoro_stream.wav \
    --stream
```

### MLX + Streaming

```bash
uv run python main.py --model kokoro \
    --text "MLX backend with streaming" \
    --output outputs/kokoro_mlx_stream.wav \
    --use-mlx \
    --stream
```

**Note**: MLX backend currently doesn't support streaming playback during generation (coming soon).

## Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--text` | Required | Any text | Text to synthesize |
| `--output` | `output.wav` | Path | Output file |
| `--voice` | `af_heart` | See voices | Voice style |
| `--lang-code` | `e` | a,b,e,f,i,j,p,z | Language code |
| `--speed` | `1.0` | 0.5-2.0 | Speech speed |
| `--stream` | False | Flag | Enable streaming |
| `--use-mlx` | False | Flag | Use MLX backend (Apple Silicon) |run python main.py --model kokoro \
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
| `h` | Hindi | `d` | German |
| `r` | Russian | `t` | Turkish |

**Note for Japanese (`j`)**: 
Requires `unidic` dictionary. Run:
```bash
uv run python -m unidic download
```

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
