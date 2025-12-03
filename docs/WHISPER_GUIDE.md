# Whisper ASR - Usage Guide

**OpenAI Whisper** is a robust, multilingual ASR model capable of transcription and translation. It is highly resistant to background noise and accents.

## Features

- **Multilingual**: Supports 99 languages.
- **Translation**: Translates any supported language **to English**.
- **Timestamps**: Provides word/segment-level timestamps.
- **Robustness**: Excellent performance on noisy audio.

## Usage Examples

### 1. Basic Transcription (Auto-Detect Language)
```bash
uv run python main.py --model whisper \
  --reference samples/speech.wav \
  --output outputs/transcript.txt
```

### 2. Transcription with Timestamps
Include timing information in the output.
```bash
uv run python main.py --model whisper \
  --reference samples/speech.wav \
  --timestamps \
  --output outputs/transcript_timed.txt
```

### 3. Specify Source Language
Improves accuracy if the language is known (e.g., French).
```bash
uv run python main.py --model whisper \
  --reference samples/french.wav \
  --language fr \
  --output outputs/french.txt
```

### 4. Translate to English
Translate foreign speech (e.g., Spanish) to English text.
```bash
uv run python main.py --model whisper \
  --reference samples/spanish.wav \
  --language es \
  --target-language en \
  --output outputs/translation.txt
```

### 5. Select Specific Model
Use `--model-id` to specify any HuggingFace Whisper model.
```bash
# Use Large v3 Turbo (Faster, high accuracy)
uv run python main.py --model whisper \
  --reference samples/speech.wav \
  --model-id openai/whisper-large-v3-turbo \
  --output outputs/turbo.txt

# Use Medium (Balanced)
uv run python main.py --model whisper \
  --reference samples/speech.wav \
  --model-id openai/whisper-medium \
  --output outputs/medium.txt
```

### 6. Apple Silicon Optimization (MLX)
Use `--use-mlx` for significantly faster performance on Mac (M1/M2/M3).
```bash
# Uses mlx-community/whisper-large-v3-turbo by default
uv run python main.py --model whisper \
  --reference samples/speech.wav \
  --use-mlx \
  --output outputs/mlx_turbo.txt

# Use Medium with MLX (Fast)
uv run python main.py --model whisper \
  --reference samples/speech.wav \
  --use-mlx \
  --model-id mlx-community/whisper-medium \
  --output outputs/mlx_medium.txt
```

## Supported Languages

Whisper supports a vast number of languages including:
English (en), Chinese (zh), German (de), Spanish (es), Russian (ru), Korean (ko), French (fr), Japanese (ja), Portuguese (pt), Turkish (tr), Polish (pl), Catalan (ca), Dutch (nl), Arabic (ar), Swedish (sv), Italian (it), Indonesian (id), Hindi (hi), Finnish (fi), Vietnamese (vi), Hebrew (he), Ukrainian (uk), Greek (el), Malay (ms), Czech (cs), Romanian (ro), Danish (da), Hungarian (hu), Tamil (ta), Norwegian (no), Thai (th), Urdu (ur), Croatian (hr), Bulgarian (bg), Lithuanian (lt), Latin (la), Maori (mi), Malayalam (ml), Welsh (cy), Slovak (sk), Telugu (te), Persian (fa), Latvian (lv), Bengali (bn), Serbian (sr), Azerbaijani (az), Slovenian (sl), Kannada (kn), Estonian (et), Macedonian (mk), Breton (br), Basque (eu), Icelandic (is), Armenian (hy), Nepali (ne), Mongolian (mn), Bosnian (bs), Kazakh (kk), Albanian (sq), Swahili (sw), Galician (gl), Marathi (mr), Punjabi (pa), Sinhala (si), Khmer (km), Shona (sn), Yoruba (yo), Somali (so), Afrikaans (af), Occitan (oc), Georgian (ka), Belarusian (be), Tajik (tg), Sindhi (sd), Gujarati (gu), Amharic (am), Yiddish (yi), Lao (lo), Uzbek (uz), Faroese (fo), Haitian Creole (ht), Pashto (ps), Turkmen (tk), Nynorsk (nn), Maltese (mt), Sanskrit (sa), Luxembourgish (lb), Myanmar (my), Tibetan (bo), Tagalog (tl), Malagasy (mg), Assamese (as), Tatar (tt), Hawaiian (haw), Lingala (ln), Hausa (ha), Bashkir (ba), Javanese (jw), Sundanese (su).

## Configuration

- **Model Selection**:
  - Default (GPU/MPS): `openai/whisper-large-v3`
  - Default (CPU): `openai/whisper-tiny`
  - Default (MLX): `mlx-community/whisper-large-v3-turbo`
  - Override with `--model-id` argument.

- **MLX Optimization**:
  - Use `--use-mlx` to enable `mlx-whisper` backend.
  - Requires `mlx-whisper` package (installed automatically).
  - Much faster on Apple Silicon.

## Accuracy Notes

**Benchmark Results (Kokoro TTS-synthesized audio):**
- **Spanish**: 100% accuracy (0% CER) - Perfect! ✅
- **English**: 80-85% accuracy (15-20% CER) - Some errors ⚠️

**Key Insight:** The accuracy difference suggests Kokoro TTS produces clearer Spanish than English, or has pronunciation artifacts in English synthesis. For production use:
- ✅ Test with real recorded audio for accurate assessment
- ⚠️ Synthesized test audio may not reflect real-world performance
- ✅ All Whisper models achieved perfect Spanish transcription
