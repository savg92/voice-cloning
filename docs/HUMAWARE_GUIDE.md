# HumAware VAD - Usage Guide

**HumAware VAD** is a fine-tuned Voice Activity Detection model based on Silero VAD. It is specifically designed to distinguish between **speech** and **humming/non-speech vocalizations**.

## Features

- **Robust Detection**: Distinguishes speech from humming.
- **Adjustable Sensitivity**: Control detection threshold.
- **Fine-Grained Control**: Adjust minimum speech/silence durations and padding.
- **Fast**: Runs efficiently on CPU/GPU.

## Usage Examples

### 1. Basic Speech Detection
Detect speech segments using default settings.
```bash
uv run python main.py --model humaware \
  --reference samples/audio.wav \
  --output outputs/segments.txt
```

### 2. Adjust Sensitivity
Use `--vad-threshold` to change sensitivity (0.0 - 1.0). Higher values mean stricter detection (less false positives).
```bash
uv run python main.py --model humaware \
  --reference samples/audio.wav \
  --vad-threshold 0.7 \
  --output outputs/segments_strict.txt
```

### 3. Fine-Tune Durations
- `--min-speech-ms`: Ignore speech chunks shorter than this (default: 250ms).
- `--min-silence-ms`: Ignore silence chunks shorter than this (default: 100ms).
- `--speech-pad-ms`: Add padding to detected speech chunks (default: 30ms).

```bash
uv run python main.py --model humaware \
  --reference samples/audio.wav \
  --min-speech-ms 500 \
  --min-silence-ms 200 \
  --output outputs/segments_tuned.txt
```

## Output Format

The output file contains start and end timestamps (in seconds) for each detected speech segment:
```text
0.50 - 2.30
3.10 - 5.45
...
```
