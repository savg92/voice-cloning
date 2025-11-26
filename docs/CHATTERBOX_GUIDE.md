# Chatterbox TTS - Installation & Usage

## ⚠️ Dependency Conflict

**Important:** Chatterbox TTS has a strict dependency conflict with other models:
- `chatterbox-tts` requires `transformers==4.46.3` (hardcoded in package)
- `mlx-audio` (Marvis TTS) requires `transformers>=4.49.0`

**Investigation Findings:**
- Chatterbox's source code does NOT directly use `transformers.AutoModel` or `transformers.AutoTokenizer`
- It uses custom model classes and loads weights via `safetensors`
- The strict `transformers==4.46.3` constraint appears unnecessary forChatterbox's core functionality
- However, the PyPI package has this hardcoded and cannot be bypassed via dependency resolution

**Recommendation:** Use separate environment (see Option 1 below) or use Marvis TTS which also supports voice cloning without conflicts.

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

###CFG Weight Control
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
| `--exaggeration` | float | 0.5 | Emotion/intensity control (0-1) |
| `--cfg-weight` | float | 0.5 | CFG guidance weight (0-1) |
| `--multilingual` | flag | false | Use multilingual model |
| `--language` | string | optional | Language code (e.g., 'fr', 'zh', 'ja') |

## Tips

**General Use:**
- Default settings (`exaggeration=0.5`, `cfg=0.5`) work well for most cases
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

1. **Dependency Conflict:** Cannot be installed alongside mlx-audio due to transformers version mismatch
2. **Recommendation:** Use separate virtual environment if you need both Chatterbox and Marvis/Granite models
3. **Alternative:** Use Marvis TTS which also supports voice cloning and is compatible with other models
