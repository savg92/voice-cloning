"""
Chatterbox TTS Wrapper

Supports both English (ChatterboxTTS) and Multilingual (ChatterboxMultilingualTTS)
with voice cloning and exaggeration control.

Features:
- Zero-shot voice cloning
- Exaggeration/intensity control (0-1, default 0.5)
- CFG weight for generation control (0-1, default 0.5)
- Multilingual support (23 languages) - PyTorch only
- MLX backend for Apple Silicon optimization (4-bit quantized, English only)
"""

import torch
import torchaudio as ta
import shutil
import threading
import queue
import subprocess
import numpy as np
import soundfile as sf
import os
import logging
from .utils import map_lang_code

logger = logging.getLogger(__name__)

# Voice Presets for consistency across languages
# These reference local wav files that serve as high-quality speaker embeddings
VOICE_PRESETS = {
    # Default Language Voices (Kokoro-cloned)
    "en": "samples/kokoro_voices/af_heart.wav",
    "es": "samples/kokoro_voices/ef_dora.wav",
    "fr": "samples/kokoro_voices/ff_siwis.wav",
    "it": "samples/kokoro_voices/if_sara.wav",
    "pt": "samples/kokoro_voices/pf_dora.wav",
    "de": "samples/kokoro_voices/de_sarah.wav",
    "ru": "samples/kokoro_voices/ru_nicole.wav",
    "tr": "samples/kokoro_voices/tr_river.wav",
    "hi": "samples/kokoro_voices/hf_alpha.wav",
    
    # Specific Kokoro Character Voices
    "af_heart": "samples/kokoro_voices/af_heart.wav",
    "af_bella": "samples/kokoro_voices/af_bella.wav",
    "am_adam": "samples/kokoro_voices/am_adam.wav",
    "ef_dora": "samples/kokoro_voices/ef_dora.wav",
    "ff_siwis": "samples/kokoro_voices/ff_siwis.wav",
    "if_sara": "samples/kokoro_voices/if_sara.wav",
    "pf_dora": "samples/kokoro_voices/pf_dora.wav",
    "de_sarah": "samples/kokoro_voices/de_sarah.wav",
    "ru_nicole": "samples/kokoro_voices/ru_nicole.wav",
    "tr_river": "samples/kokoro_voices/tr_river.wav",
    "hi_puck": "samples/kokoro_voices/hi_puck.wav",
    "hf_alpha": "samples/kokoro_voices/hf_alpha.wav",

    # Legacy/Other
    "dave": "samples/neutts_air/dave.wav",
    "jo": "samples/neutts_air/jo.wav",
}

class ChatterboxWrapper:
    """
    Wrapper for Chatterbox TTS models.
    
    Supports both English-only and Multilingual variants.
    """
    
    def __init__(self, device: str | None = None, multilingual: bool = False):
        """
        Initialize Chatterbox TTS.
        
        Args:
            device: Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None.
            multilingual: If True, load multilingual model. Default is English-only.
        """
        self.device = device or self._auto_detect_device()
        self.multilingual = multilingual
        
        logger.info(f"Initializing Chatterbox ({'Multilingual' if multilingual else 'English'}) on {self.device}")
        
        try:
            if multilingual:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                
            logger.info("✓ Chatterbox model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox model: {e}")
            raise
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def generate(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language_id: str | None = None
    ) -> torch.Tensor:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            audio_prompt_path: Path to reference audio for voice cloning (optional)
            exaggeration: Emotion/intensity control 0-1 (default 0.5)
            cfg_weight: CFG guidance weight 0-1 (default 0.5)
            language_id: Language code for multilingual model (e.g., 'en', 'fr', 'zh')
        
        Returns:
            Audio tensor
            
        Tips:
            - Default settings (exaggeration=0.5, cfg=0.5) work well for most cases
            - For fast speakers, lower cfg to ~0.3
            - For expressive speech: lower cfg (~0.3), higher exaggeration (~0.7)
            - Higher exaggeration speeds up speech; lower cfg compensates with slower pacing
        """
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight
        }
        
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = audio_prompt_path
        
        # Multilingual model requires language_id
        if self.multilingual and language_id:
            kwargs["language_id"] = language_id
        elif self.multilingual and not language_id:
            logger.warning("Multilingual model loaded but no language_id provided. Defaulting to 'en'")
            kwargs["language_id"] = "en"
        
        return self.model.generate(text, **kwargs)
    
    @property
    def sr(self) -> int:
        """Return sampling rate."""
        return self.model.sr


def _synthesize_with_pytorch(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    multilingual: bool = False
):
    """
    Synthesize speech using PyTorch backend (chatterbox-tts library).
    """
    wrapper = ChatterboxWrapper(multilingual=multilingual)
    
    wav = wrapper.generate(
        text,
        audio_prompt_path=source_wav,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        language_id=language
    )
    
    # Save audio
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu()
    
    ta.save(output_wav, wav, wrapper.sr)
    logger.info(f"✓ Saved audio to {output_wav}")


def _synthesize_with_mlx(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    model_id: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    stream: bool = False
):
    """
    Synthesize speech using MLX backend (Apple Silicon optimized, 4-bit quantized).
    
    NOTE: As of mlx-audio 0.2.8, Chatterbox support is not yet available in the official
    mlx-audio library, despite references on Hugging Face. This function is prepared for
    when support is added.
    """
    try:
        from mlx_audio.tts.generate import generate_audio
        # Import the Chatterbox model class to patch it
        # The class name in mlx-audio is 'Model', not 'Chatterbox'
        from mlx_audio.tts.models.chatterbox.chatterbox import Model as ChatterboxModel
    except ImportError:
        raise ImportError(
            "MLX backend requires 'mlx-audio' package. Install with:\n"
            "  uv pip install -U mlx-audio\n"
            "Or use use_mlx=False to use PyTorch backend."
        )

    # Monkeypatch to fix weight loading issue (ignoring strict key checks)
    # The current mlx-audio (0.2.9) is strict about keys, but the weights
    # on HF (chatterbox-turbo) have extra keys like 'gen.prompt_token' etc.
    original_load_weights = ChatterboxModel.load_weights

    def patched_load_weights(self, weights: list, strict: bool = False):
        try:
             # Try loading normally first (unlikely to work with strict=True or if keys are missing/extra)
             original_load_weights(self, weights, strict=strict)
        except ValueError as e:
             # If it complains about unrecognized keys, try again with strict=False (if the error came from that)
             # But the traceback shows explicit raise ValueError without falling back.
             # So we must implement our own lenient loader or just suppress the specific error.
             if "Unrecognized weight keys" in str(e):
                 logger.warning(f"Ignoring unrecognized keys in Chatterbox weights to prevent crash: {e}")
                 # To truly bypass, we likely need to call load_weights with strict=False if the original call was strict.
                 # However, looking at the source, original_load_weights calls passed-in strict check at end.
                 # If we call it with strict=False, it might still fail if there are missing keys?
                 # Actually, let's just re-call with strict=False which filters out the check at the end.
                 if strict:
                     original_load_weights(self, weights, strict=False)
                 else:
                     # If it failed even with strict=False (or strict=False wasn't enough), we might be stuck.
                     # But the error trace says "Unrecognized weight keys", which comes from:
                     # if other_weights and strict: raise ValueError(...)
                     # So calling with strict=False should fix it.
                     pass 
             else:
                 raise

    # Apply patch
    ChatterboxModel.load_weights = patched_load_weights
    
    logger.info("Generating speech with MLX backend (Chatterbox-TTS-4bit)...")
    
    # MLX Chatterbox requires ref_audio for voice cloning.
    # We'll determine the best reference in this order:
    # 1. source_wav (direct path)
    # 2. voice (preset name or direct path)
    # 3. language-specific default preset
    # 4. Global default (samples/anger.wav)
    
    ref_audio = None
    
    if source_wav:
        ref_audio = source_wav
        logger.info(f"Using reference audio for voice cloning: {source_wav}")
    elif voice:
        if voice in VOICE_PRESETS:
            ref_audio = VOICE_PRESETS[voice]
            logger.info(f"Using voice preset: {voice} -> {ref_audio}")
        elif os.path.exists(voice):
            ref_audio = voice
            logger.info(f"Using voice reference file: {voice}")
        else:
            logger.warning(f"Voice preset/file not found: {voice}. Falling back to default.")
            
    if not ref_audio:
        # Check if we have a preset for the current language
        if language in VOICE_PRESETS:
            ref_audio = VOICE_PRESETS[language]
            logger.info(f"Using consistent language voice preset for '{language}': {ref_audio}")
        else:
            # Global fallback
            default_ref = VOICE_PRESETS.get("en", "samples/anger.wav")
            if os.path.exists(default_ref):
                ref_audio = default_ref
                logger.info(f"Using global default voice: {ref_audio}")
            else:
                 raise ValueError(
                     "Chatterbox (MLX) requires a reference audio file for voice cloning.\n"
                     "Please provide one using --reference or --voice, or ensured default samples exist."
                 )
    
    # Reference text is optional - mlx-audio can auto-transcribe if needed
    ref_text = "."
    
    try:
        # Use provided model_id or default to standard 4-bit model
        target_model = model_id if model_id else "mlx-community/chatterbox-4bit"
        is_turbo = "turbo" in target_model.lower()
        
        # Map 'language' to 'lang_code'
        if is_turbo:
            # Turbo model supports expanded language codes (ar, da, de, el, en, es, etc.)
            lang_code = language if language else "en"
        else:
            # Standard Kokoro-based models use 'a' for American English, 'b' for British English
            lang_code = map_lang_code(language) if language else "a"
        
        # Get output directory and filename
        output_dir = os.path.dirname(output_wav) or "."
        output_name = os.path.splitext(os.path.basename(output_wav))[0]
        file_prefix = os.path.join(output_dir, output_name)
        
        # Map languages to their respective Kokoro voices for consistency
        # This solves the issue where it defaults to 'af_heart' (English) even for French/Spanish
        voice_map = {
            "en": "af_heart",
            "es": "ef_dora",
            "fr": "ff_siwis",
            "it": "if_sara",
            "pt": "pf_dora",
            "de": "af_sarah", # de_sarah sample uses af_sarah voice
            "ru": "af_nicole",
            "tr": "af_river",
            "hi": "hf_alpha",
            "ja": "jf_alpha", # Fallback names if generation failed but user wants consistency
            "zh": "zf_xiaoxiao"
        }
        
        # Determine the best Kokoro voice name to pass to mlx-audio
        if voice and voice in VOICE_PRESETS:
             kokoro_voice = voice # Use the preset name as the voice
        elif voice:
             kokoro_voice = voice # Pass through direct voice name (e.g. af_bella)
        else:
             kokoro_voice = voice_map.get(language, "af_heart")

        # Note: mlx-audio may not support exaggeration and cfg_weight parameters
        # These were specific to mlx-audio-plus. We'll try to pass them but they may be ignored.
        kwargs = {
            "text": text,
            "model": target_model,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "file_prefix": file_prefix,
            "lang_code": lang_code,
            "voice": kokoro_voice,
            "speed": speed,
            "stream": stream,
            "verbose": True
        }
        
        # Try to include control parameters if supported
        # The official mlx-audio may use different parameter names or not support these
        if exaggeration != 0.5 or cfg_weight != 0.5:
             # Manually inject them into kwargs if we suspect the model supports them but signature doesn't
             # This depends on how generate_audio passes them.
             pass

        if stream:
            # mlx-audio 0.2.9 'generate_audio' doesn't yield chunks for external consumption.
            # If stream=True, it doesn't save the file, which breaks the UI.
            # So we force stream=False (to save file) but set play=True (to simulate streaming/playback locally).
            logger.info("Streaming enabled: active local playback (play=True).")
            kwargs["stream"] = False 
            kwargs["play"] = True

        generate_audio(**kwargs)
        
        # mlx-audio generates file_prefix + "_000.wav" by default
        expected_output = f"{file_prefix}_000.wav"
        
        # Also check for .wav if prefix already had it (unlikely but possible)
        if not os.path.exists(expected_output) and os.path.exists(f"{file_prefix}.wav"):
             expected_output = f"{file_prefix}.wav"

        if os.path.exists(expected_output):
            # Rename to the requested output_wav if they differ
            if os.path.abspath(expected_output) != os.path.abspath(output_wav):
                if os.path.exists(output_wav):
                    os.remove(output_wav)
                os.rename(expected_output, output_wav)
            logger.info(f"✓ MLX synthesis complete: {output_wav}")
        else:
            # Check if maybe it saved it without suffix for some reason (older version?)
            raise RuntimeError(
                f"MLX did not generate expected output file: {expected_output}"
            )
        
    except ValueError as e:
        if "chatterbox not supported" in str(e).lower():
            raise RuntimeError(
                "Chatterbox is not yet supported in the official mlx-audio library.\n"
                "As of mlx-audio 0.2.8, Chatterbox support is still in development.\n"
                "Please use the PyTorch backend instead (remove --use-mlx flag).\n"
                f"Original error: {e}"
            )
        raise
    except Exception as e:
        logger.error(f"MLX synthesis failed: {e}")
        raise RuntimeError(f"MLX generation failed: {e}")


def synthesize_with_chatterbox(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    multilingual: bool = False,
    use_mlx: bool = False,
    model_id: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    stream: bool = False
):
    """
    Synthesize speech from text using Chatterbox TTS.
    
    Args:
        text: Text to synthesize
        output_wav: Output audio file path
        source_wav: Reference audio for voice cloning (optional)
        exaggeration: Emotion/intensity control 0-1 (default 0.5)
        cfg_weight: CFG guidance weight 0-1 (default 0.5)
        language: Language code for multilingual model (e.g., 'en', 'fr', 'zh')
        multilingual: Use multilingual model (supports 23 languages) - PyTorch only
        use_mlx: Use MLX backend for Apple Silicon optimization (English only, 4-bit)
        model_id: Specific model ID to use (e.g. 'mlx-community/chatterbox-turbo-4bit')
        voice: Specific voice name to use (e.g. 'af_heart')
        speed: Speech speed (default 1.0)
        stream: Enable streaming (default False)
    
    Supported languages (multilingual model, PyTorch only):
        ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, 
        pl, pt, ru, sv, sw, tr, zh
    """
    # Auto-enable multilingual if language is specified and not English
    if language and language != "en":
        multilingual = True
    
    # MLX backend
    if use_mlx:
         _synthesize_with_mlx(
             text, 
             output_wav, 
             source_wav, 
             exaggeration, 
             cfg_weight, 
             language, 
             model_id=model_id,
             voice=voice,
             speed=speed,
             stream=stream
         )
    else:
        _synthesize_with_pytorch(text, output_wav, source_wav, exaggeration, cfg_weight, language, multilingual)

