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

## Supported Languages

Whisper supports a vast number of languages including:
English (en), Chinese (zh), German (de), Spanish (es), Russian (ru), Korean (ko), French (fr), Japanese (ja), Portuguese (pt), Turkish (tr), Polish (pl), Catalan (ca), Dutch (nl), Arabic (ar), Swedish (sv), Italian (it), Indonesian (id), Hindi (hi), Finnish (fi), Vietnamese (vi), Hebrew (he), Ukrainian (uk), Greek (el), Malay (ms), Czech (cs), Romanian (ro), Danish (da), Hungarian (hu), Tamil (ta), Norwegian (no), Thai (th), Urdu (ur), Croatian (hr), Bulgarian (bg), Lithuanian (lt), Latin (la), Maori (mi), Malayalam (ml), Welsh (cy), Slovak (sk), Telugu (te), Persian (fa), Latvian (lv), Bengali (bn), Serbian (sr), Azerbaijani (az), Slovenian (sl), Kannada (kn), Estonian (et), Macedonian (mk), Breton (br), Basque (eu), Icelandic (is), Armenian (hy), Nepali (ne), Mongolian (mn), Bosnian (bs), Kazakh (kk), Albanian (sq), Swahili (sw), Galician (gl), Marathi (mr), Punjabi (pa), Sinhala (si), Khmer (km), Shona (sn), Yoruba (yo), Somali (so), Afrikaans (af), Occitan (oc), Georgian (ka), Belarusian (be), Tajik (tg), Sindhi (sd), Gujarati (gu), Amharic (am), Yiddish (yi), Lao (lo), Uzbek (uz), Faroese (fo), Haitian Creole (ht), Pashto (ps), Turkmen (tk), Nynorsk (nn), Maltese (mt), Sanskrit (sa), Luxembourgish (lb), Myanmar (my), Tibetan (bo), Tagalog (tl), Malagasy (mg), Assamese (as), Tatar (tt), Hawaiian (haw), Lingala (ln), Hausa (ha), Bashkir (ba), Javanese (jw), Sundanese (su).

## Configuration

- **Model Size**: Defaults to `large-v3` on GPU/MPS, `tiny` on CPU.
- **Environment Variable**: Override model size with `WHISPER_MODEL` (e.g., `export WHISPER_MODEL=openai/whisper-medium`).
