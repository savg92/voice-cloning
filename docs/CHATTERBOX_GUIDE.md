# Chatterbox TTS - Installation & Usage

## ⚠️ Dependency Conflict

**Important:** Chatterbox TTS has a strict dependency conflict with other models:
- `chatterbox-tts` requires `transformers==4.46.3` (hardcoded in package)
- `mlx-audio` (Marvis TTS) requires `transformers>=4.49.0`

**MLX Optimized Version:**
You can use the MLX-optimized 4-bit model with the official `mlx-audio` package.

---

## 🍎 MLX Backend Support (Apple Silicon)

**Status:** ⏳ **PENDING** (Chatterbox not yet in mlx-audio 0.2.8)

The Hugging Face model page for `mlx-community/chatterbox-4bit` references `mlx-audio`, but as of version 0.2.8, Chatterbox support is not yet implemented in the official library. However, we've implemented support for both the standard and **turbo** versions via the `--model-id` flag.

### Turbo Model

The turbo model (`mlx-community/chatterbox-turbo-4bit`) is significantly faster and optimized for real-time performance.


### Prerequisites

Install the MLX Audio library:

```bash
uv pip install -U mlx-audio
```

### Usage

Use the `--use-mlx` flag to enable the MLX backend. 

**Note:** The MLX backend supports both standard synthesis (default voice) and voice cloning.

```bash
# Basic Usage (uses default fallback reference if none provided)
uv run python main.py --model chatterbox \
  --text "Running on Apple Silicon with MLX 4-bit quantization." \
  --use-mlx \
  --output outputs/mlx_chatterbox.wav

# Voice Cloning (Recommended)
uv run python main.py --model chatterbox \
  --text "Cloning this voice on MLX." \
  --reference samples/anger.wav \
  --use-mlx \
  --output outputs/cloned_mlx.wav

# Using Chatterbox Turbo (Fastest)
uv run python main.py --model chatterbox \
  --text "Testing the new turbo model for faster inference." \
  --use-mlx \
  --model-id mlx-community/chatterbox-turbo-4bit \
  --output outputs/turbo_test.wav

# Multilingual Support (Turbo only)
# Languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
uv run python main.py --model chatterbox \
  --text "¡Hola! Estoy probando el soporte multilingüe de Chatterbox Turbo." \
  --language es \
  --use-mlx \
  --model-id mlx-community/chatterbox-turbo-4bit \
  --output outputs/turbo_spanish.wav
```

### Supported Languages (Turbo Model)
The `chatterbox-turbo-4bit` model supports the following languages:

| Language | Code | Language | Code |
| :--- | :--- | :--- | :--- |
| Arabic | `ar` | Italian | `it` |
| Chinese | `zh` | Japanese | `ja` |
| Danish | `da` | Korean | `ko` |
| Dutch | `nl` | Malay | `ms` |
| English | `en` | Norwegian | `no` |
| Finnish | `fi` | Polish | `pl` |
| French | `fr` | Portuguese | `pt` |
| German | `de` | Russian | `ru` |
| Greek | `el` | Spanish | `es` |
| Hebrew | `he` | Swahili | `sw` |
| Hindi | `hi` | Swedish | `sv` |
| | | Turkish | `tr` |

### Voice Presets (Kokoro Consistency)
For consistency across languages, you can use these high-quality Kokoro voice presets. They are automatically mapped to the appropriate language if no other reference is provided:

| Preset Name | Recommended Language | Description |
| :--- | :--- | :--- |
| `af_heart` | `en` | Standard high-quality female voice (Global default) |
| `af_bella` | `en` | Expressive female voice |
| `am_adam` | `en` | Deep male voice |
| `ef_dora` | `es` | Spanish-optimized female voice |
| `ff_siwis` | `fr` | French-optimized female voice |
| `if_sara` | `it` | Italian-optimized female voice |
| `pf_dora` | `pt` | Portuguese-optimized female voice |

Example usage:
```bash
# Uses the French-optimized Kokoro voice for consistency
uv run python main.py --model chatterbox --text "C'est une voix française consistante." --language fr --use-mlx --model-id mlx-community/chatterbox-turbo-4bit --output outputs/turbo_fr_consistant.wav

# Explicitly selecting a Kokoro voice
uv run python main.py --model chatterbox --text "Using the Adam voice for English." --voice am_adam --use-mlx --model-id mlx-community/chatterbox-turbo-4bit --output outputs/turbo_adam.wav
```


**Features:**
- 4-bit quantized model (low memory)
- Fast inference on M-series chips
- Voice cloning support
- **Multilingual Support:** 23 languages (same as PyTorch version)
- **Control:** Exaggeration, Speed (via cfg), and more

### MLX Multilingual Usage
```bash
# Spanish
uv run python main.py --model chatterbox \
  --text "Hola, esto es una prueba en español." \
  --language es \
  --use-mlx \
  --reference samples/anger.wav \
  --output outputs/mlx_spanish.wav
```

### MLX Control Parameters
```bash
# Expressive speech
uv run python main.py --model chatterbox \
  --text "This is very dramatic!" \
  --exaggeration 0.8 \
  --cfg-weight 0.4 \
  --use-mlx \
  --reference samples/anger.wav \
  --output outputs/mlx_dramatic.wav
```

> [!WARNING]
> **MLX Backend Status (December 2024):** While the Hugging Face model page for `mlx-community/chatterbox-4bit` suggests support in `mlx-audio`, as of version 0.2.8, Chatterbox is **not yet implemented** in the official `mlx-audio` library. The module `mlx_audio.tts.models.chatterbox` does not exist. 
>
> **Recommendation:** Use the PyTorch backend (without `--use-mlx` flag) for Chatterbox TTS until official MLX support is added.


## Installation Options

### Option 1: Separate Virtual Environment (Recommended)

Create a separate environment for Chatterbox:

**Using `uv` (Recommended):**

```bash
# Create separate project directory for Chatterbox
mkdir chatterbox_env
cd chatterbox_env

# Create a new uv project
uv init

# Add chatterbox-tts dependency
uv add chatterbox-tts torch torchaudio

# Use it
uv run python -c "
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

model = ChatterboxTTS.from_pretrained(device='cuda')
text = 'Hello from Chatterbox!'
wav = model.generate(text)
ta.save('output.wav', wav, model.sr)
"
```

**Using standard venv:**

```bash
# Create separate venv for Chatterbox
python -m venv chatterbox_env
source chatterbox_env/bin/activate  # On Windows: chatterbox_env\Scripts\activate

# Install chatterbox-tts
pip install chatterbox-tts torch torchaudio

# Use it
python -c "
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

model = ChatterboxTTS.from_pretrained(device='cuda')
text = 'Hello from Chatterbox!'
wav = model.generate(text)
ta.save('output.wav', wav, model.sr)
"
```

### Option 2: Use with This Project (May Break Other Models)

If you only need Chatterbox and can accept breaking Marvis TTS:

```bash
uv pip install chatterbox-tts
```

**Warning:** This will downgrade `transformers` to 4.46.3 and may break:
- Marvis TTS (mlx-audio)
- Granite ASR
- Other models requiring newer transformers

## Features

### English TTS
```bash
uv run python main.py --model chatterbox \
  --text "Hello from Chatterbox!" \
  --output outputs/chatterbox.wav
```

### Voice Cloning
```bash
uv run python main.py --model chatterbox \
  --text "Clone this voice" \
  --reference samples/anger.wav \
  --output outputs/cloned.wav
```

### Exaggeration Control
```bash
# Subtle (0.3)
uv run python main.py --model chatterbox \
  --text "Calm and measured speech" \
  --exaggeration 0.3 \
  --output outputs/calm.wav

# Expressive (0.8)
uv run python main.py --model chatterbox \
  --text "Dramatic and expressive!" \
  --exaggeration 0.8 \
  --output outputs/dramatic.wav
```

### CFG Weight Control
```bash
# Lower CFG for faster pacing
uv run python main.py --model chatterbox \
  --text "Fast speech" \
  --cfg-weight 0.3 \
  --output outputs/fast.wav
```

### Multilingual (23 Languages)
```bash
# French
uv run python main.py --model chatterbox \
  --text "Bonjour, comment ça va?" \
  --language fr \
  --multilingual \
  --output outputs/french.wav

# Chinese
uv run python main.py --model chatterbox \
  --text "你好，今天天气真不错" \
  --language zh \
  --multilingual \
  --output outputs/chinese.wav

# Japanese
uv run python main.py --model chatterbox \
  --text "こんにちは、元気ですか？" \
  --language ja \
  --multilingual \
  --output outputs/japanese.wav
```

## Supported Languages (Multilingual Model)

Arabic (ar), Danish (da), German (de), Greek (el), English (en), Spanish (es), 
Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja), 
Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), 
Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--text` | string | required | Text to synthesize |
| `--output` | string | required | Output audio file path |
| `--reference` | string | optional | Reference audio for voice cloning |
| `--exaggeration` | float | 0.7 | Emotion/intensity control (0-1) |
| `--cfg-weight` | float | 0.5 | CFG guidance weight (0-1) |
| `--multilingual` | flag | false | Use multilingual model |
| `--language` | string | optional | Language code (e.g., 'fr', 'zh', 'ja') |

## Tips

**General Use:**
- Default settings (`exaggeration=0.7`, `cfg=0.5`) work well for most cases
- For fast speakers, lower `cfg` to ~0.3

**Expressive/Dramatic Speech:**
- Lower `cfg` (~0.3) and increase `exaggeration` (~0.7+)
- Higher exaggeration speeds up speech; lower cfg compensates with slower pacing

**Language Transfer:**
- Ensure reference clip matches the specified language tag
- Otherwise, output may inherit reference clip's accent
- To mitigate: set `--cfg-weight 0`

## Direct Python API

```python
from chatterbox.tts import ChatterboxTTS
import torchaudio as ta

# English model
model = ChatterboxTTS.from_pretrained(device="cuda")
wav = model.generate(
    "Hello from Chatterbox!",
    exaggeration=0.7,
    cfg_weight=0.5
)
ta.save("output.wav", wav, model.sr)

# Voice cloning
wav = model.generate(
    "Clone this voice",
    audio_prompt_path="reference.wav",
    exaggeration=0.5,
    cfg_weight=0.5
)
ta.save("cloned.wav", wav, model.sr)
```

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torchaudio as ta

# Multilingual model
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# French
wav_fr = model.generate(
    "Bonjour, comment ça va?",
    language_id="fr",
    exaggeration=0.5
)
ta.save("french.wav", wav_fr, model.sr)

# Chinese
wav_zh = model.generate(
    "你好，今天天气真不错",
    language_id="zh",
    exaggeration=0.5
)
ta.save("chinese.wav", wav_zh, model.sr)
```

## Known Limitations

1. **Dependency Conflict:** Cannot be installed alongside mlx-audio due to transformers version mismatch (Use MLX backend to avoid this!)
2. **Recommendation:** Use `--use-mlx` on Apple Silicon for best performance and compatibility.
