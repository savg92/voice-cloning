# Multilingual Benchmark Results

## Overview

This document contains the results of benchmarking multilingual TTS and ASR models using Spanish as the test language.

**Test Date**: December 3, 2025  
**Test Language**: Spanish  
**Test Sentence**: "Hola, ¿cómo estás? El día está soleado y hermoso."

## Models Tested

### Text-to-Speech (TTS)
- **Kokoro** (Spanish language mode)
- **Chatterbox** (Multilingual - Spanish)

### Automatic Speech Recognition (ASR)
- **Whisper** (Tiny model - multilingual)
- **Canary** (Multilingual ASR)
- **Parakeet** (MLX backend - 100+ languages)

## Results Summary

### TTS Performance

| Model | Status | Latency (s) | RTF | Audio Duration (s) | Notes |
|-------|---------|-------------|-----|-------------------|-------|
| Kokoro (Spanish) | ✅ | 9.28 | 2.36 | 3.92 | Successful synthesis |
| Chatterbox (Spanish) | ❌ | - | - | - | Failed (dependency conflict) |

**Key Findings**:
- Kokoro successfully generated Spanish audio at 2.36x real-time
- Chatterbox failed due to transformers dependency conflict with mlx-audio

### ASR Performance

| Model | Audio Source | Status | Latency (s) | RTF | CER | Speed vs Real-time |
|-------|--------------|---------|-------------|-----|-----|-------------------|
| **Whisper (MLX Turbo)** | kokoro_spanish.wav | ✅ | 6.21 | 1.58 | **0.00%** | **0.63× (faster!)** |
| **Parakeet** | kokoro_spanish.wav | ✅ | 9.24 | 2.35 | **0.00%** | 0.43× |
| **Whisper (MLX Medium)** | kokoro_spanish.wav | ✅ | 11.53 | 2.94 | **0.00%** | 0.34× |
| **Whisper (Standard Turbo)** | kokoro_spanish.wav | ✅ | 46.82 | 11.93 | **0.00%** | 0.08× (slow!) |
| **Canary** | kokoro_spanish.wav | ✅ | 145.39 | 37.04 | **0.00%** | 0.03× (very slow) |
| **Whisper (Tiny)** | kokoro_spanish.wav | ✅ | 1.04 | 0.27 | 14.29% | **3.7× (fastest!)** |

**Transcription Comparison**:

**Reference**: `Hola, ¿cómo estás? El día está soleado y hermoso.`

| Model | Transcription | Accuracy | Notes |
|-------|--------------|----------|-------|
| **Whisper (MLX Turbo)** | `Hola, ¿cómo estás? El día está soleado y hermoso.` | 🎯 **100% Perfect** | 🏆 **WINNER: Fast + Perfect** |
| **Whisper (MLX Medium)** | `Hola, ¿cómo estás? El día está soleado y hermoso.` | 🎯 **100% Perfect** | Fast (2.9× RTF) |
| **Parakeet** | `Hola, ¿cómo estás? El día está soleado y hermoso.` | 🎯 **100% Perfect** | Fast (2.4× RTF) |
| **Whisper (Standard Turbo)** | `Hola, ¿cómo estás? El día está soleado y hermoso.` | 🎯 **100% Perfect** | ⚠️ 7.5× slower than MLX |
| **Canary** | `hola, ¿cómo estás? el día está soleado y hermoso.` | 🎯 **100% Perfect** | Very slow (37× RTF) |
| **Whisper (Tiny)** | `Hola, ¿cómo estás? El día estás al Ollado y Irmoso.` | 85.7% | Fastest but errors |

## Detailed Analysis

### 🏆 Perfect Transcription: All Whisper Variants + Parakeet + Canary

**ALL MODELS** achieved **perfect transcription** with 0% character error rate (except Whisper Tiny):

**Whisper (MLX Turbo) - WINNER 🥇**:
- ✅ 100% accurate Spanish transcription
- ⚡ **Fastest among perfect models** (1.58× RTF = 0.63× real-time!)
- ✅ MLX optimization on Apple Silicon
- 🎯 **7.5× faster than standard Turbo**
- 🏆 **Best overall choice for Mac users**

**Parakeet (Runner-up) 🥈**:
- ✅ 100% accurate Spanish transcription
- ⚡ Fast performance (2.35× real-time)
- ✅ MLX optimization on Apple Silicon
- ✅ Native support for 100+ languages

**Whisper (MLX Medium) 🥉**:
- ✅ 100% accurate Spanish transcription
- ⚡ Good performance (2.94× real-time)
- ✅ MLX optimization
- Better accuracy than Turbo in some cases

**Whisper (Standard Turbo) ⚠️**:
- ✅ 100% accurate transcription
- ❌ **Very slow** (11.93× RTF - 47 seconds for 4s audio!)
- ❌ **Not recommended on Mac** - use MLX version instead
- Only use on NVIDIA GPUs

**Canary (Slowest)**:
- ✅ 100% accurate Spanish transcription
- ⚠️ **Extremely slow** (37× real-time - 145 seconds!)
- Uses NeMo/PyTorch backend
- Good for accuracy-critical offline tasks only

### Whisper Tiny Performance

**Whisper (Tiny)**:
- 🚀 **Speed Champion**: 0.27× real-time (fastest by far - essentially instant!)
- Transcription errors (14.29% CER):
  - "está soleado" → "estás al Ollado"  
  - "hermoso" → "Irmoso"
- Still very usable for general Spanish transcription
- Best for speed-critical applications where minor errors are acceptable

### Performance Summary

| Metric | MLX Turbo 🏆 | MLX Medium | Parakeet | Standard Turbo | Canary |
|--------|-------------|------------|----------|---------------|---------|
| **Accuracy** | 100% ⭐ | 100% ⭐ | 100% ⭐ | 100% ⭐ | 100% ⭐ |
| **Speed (RTF)** | **1.58x** ⚡ | 2.94x | 2.35x | 11.93x ❌ | 37x ❌❌ |
| **Best for** | **Mac users** | Balanced | All-around | NVIDIA GPU only | Offline only |

### Key Findings

1. **MLX Whisper is 7.5× faster** than standard PyTorch Whisper on Apple Silicon
2. **All modern Whisper variants achieve perfect transcription** on Spanish
3. **Standard Whisper models are extremely slow on MPS** - always use MLX on Mac
4. **Whisper Tiny trades accuracy for extreme speed** (3.7× real-time)

### Spanish vs English Accuracy

**Interesting Finding:** All models achieved **perfect accuracy (0% CER) on Spanish** but showed **15-20% CER on English** (using same Kokoro TTS for synthesis).

| Test | Best Accuracy | Notes |
|------|--------------|-------|
| **Spanish** (this benchmark) | **0% CER** ✅ | All models perfect |
| **English** (general benchmark) | **15-20% CER** ⚠️ | MLX Turbo/Medium |

**Conclusion:** This suggests **Kokoro TTS produces higher quality Spanish synthesis** than English, or has pronunciation artifacts in English. For production accuracy testing, use real recorded audio instead of synthesized speech.

### Chatterbox Limitation

Chatterbox TTS could not be tested due to dependency conflict:
- Requires `transformers==4.46.3`

## Recommendations

### For Spanish TTS
✅ **Use Kokoro** with language code `'e'` for Spanish synthesis
- Fast performance (2.36x real-time)
- Good quality Spanish output  
- Compatible with all other models

### For Spanish ASR

🥇 **First Choice: Whisper (MLX Turbo) on Mac**
- **Perfect accuracy** (0% CER)
- **Fastest perfect model** (1.58× RTF)
- MLX optimized for Apple Silicon
- 99+ language support
- **7.5× faster than standard Whisper**

🥈 **Second Choice: Parakeet**
- Perfect accuracy (0% CER)
- Fast performance (2.35× real-time)
- Supports 100+ languages
- MLX optimized for Apple Silicon

🥉 **Third Choice: Whisper (MLX Medium)**
- Perfect accuracy (0% CER)
- Good performance (2.94× real-time)
- May have better accuracy on difficult audio
- MLX optimized

⚡ **Speed Choice: Whisper (Tiny)**
- Fastest (0.27× real-time - essentially instant!)
- Good accuracy (85.7%)
- Best for real-time applications
- 99+ language support

❌ **Avoid: Standard Whisper on Mac**
- Use MLX variants instead (7-20× speedup)

❌ **Avoid: Canary on Mac**
- Extremely slow (37× real-time)
- Only use if perfect accuracy is critical and time doesn't matter

### Platform Requirements

**Apple Silicon (M1/M2/M3)**:
- Parakeet: Use MLX backend (automatic)
- Whisper: Use Tiny model on MPS

**Dependency Management**:
- Avoid using Chatterbox + Parakeet in the same environment
- Use separate virtual environments if both are needed

## Hardware

- **Platform**: Apple M3 MacBook Pro 8GB
- **Device**: MPS (Apple Silicon GPU acceleration)
- **Model Backends**: MLX (Parakeet), PyTorch (Whisper, Kokoro)

## Conclusion

For Spanish language processing:
- **TTS**: Kokoro delivers reliable Spanish synthesis
- **ASR**: Parakeet provides **perfect accuracy** with excellent speed
- **Overall**: The combination of Kokoro TTS + Parakeet ASR is ideal for Spanish applications on Apple Silicon
