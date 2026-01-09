import os

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # Kept for reference, but handled in main.py or globally if needed.
# For now, we trust the previous fix for OOM.

import torch
import torchaudio as ta
import shutil
import threading
import queue
import subprocess
import numpy as np
import soundfile as sf
import gc
import logging
from .utils import map_lang_code
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Global cache for loaded models
_MODEL_CACHE = {}

# Voice Presets
VOICE_PRESETS = {
    # 23 Language Defaults (mapped to 13 available Kokoro samples)
    "en": "samples/kokoro_voices/af_heart.wav",
    "es": "samples/kokoro_voices/ef_dora.wav",
    "fr": "samples/kokoro_voices/ff_siwis.wav",
    "it": "samples/kokoro_voices/if_sara.wav",
    "pt": "samples/kokoro_voices/pf_dora.wav",
    "de": "samples/kokoro_voices/de_sarah.wav",
    "ru": "samples/kokoro_voices/ru_nicole.wav",
    "tr": "samples/kokoro_voices/tr_river.wav",
    "hi": "samples/kokoro_voices/hf_alpha.wav",
    "ja": "samples/kokoro_voices/hf_alpha.wav",  # Fallback
    "zh": "samples/kokoro_voices/hf_alpha.wav",  # Fallback
    "ko": "samples/kokoro_voices/hf_alpha.wav",  # Fallback
    "ar": "samples/kokoro_voices/hf_alpha.wav",  # Fallback
    "da": "samples/kokoro_voices/de_sarah.wav", # Close family
    "nl": "samples/kokoro_voices/de_sarah.wav", # Close family
    "sv": "samples/kokoro_voices/de_sarah.wav",  # Close family
    "no": "samples/kokoro_voices/de_sarah.wav",  # Close family
    "fi": "samples/kokoro_voices/ru_nicole.wav", # Geographic
    "pl": "samples/kokoro_voices/ru_nicole.wav", # Slavic
    "el": "samples/kokoro_voices/if_sara.wav",   # Mediterranean
    "he": "samples/kokoro_voices/if_sara.wav",   # Mediterranean
    "ms": "samples/kokoro_voices/af_heart.wav",  # Neutral
    "sw": "samples/kokoro_voices/af_heart.wav",  # Neutral

    # Character Presets
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
    "dave": "samples/neutts_air/dave.wav",
    "jo": "samples/neutts_air/jo.wav",
}

class ChatterboxWrapper:
    """
    Wrapper for Chatterbox TTS models using official high-level APIs.
    Hybrid backend: Turbo for English speed, Multilingual for 23 languages.
    """
    
    def __init__(self, device: str | None = None, model_type: str = "chatterbox-turbo", multilingual: bool = False):
        self.device = device or self._auto_detect_device()
        self.model_type = model_type
        # We store models lazily to save VRAM/memory
        self.models = {} 
        
        logger.info(f"Initializing Hybrid {model_type} on {self.device}")

        # BROAD MONKEYPATCHES for MPS and CPU-only (CUDA deserialization) compatibility
        try:
            # 1. Patch torch.load GLOBALLY during Chatterbox initialization to fix the CUDA deserialization error
            # This is safer than just patching mtl_tts because other sub-modules might have it too
            import torch
            _orig_torch_load = torch.load
            def _safe_torch_load(f, *args, **kwargs):
                if 'map_location' not in kwargs:
                    # If we aren't on CUDA, default to CPU for loading (PyTorch will move it later)
                    if not torch.cuda.is_available():
                        kwargs['map_location'] = 'cpu'
                return _orig_torch_load(f, *args, **kwargs)
            torch.load = _safe_torch_load

            # 2. Patch S3Tokenizer (float64 avoids)
            from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer
            _orig_prepare = S3Tokenizer._prepare_audio
            def _patched_prepare(self_tok, wavs):
                processed = _orig_prepare(self_tok, wavs)
                return [w.to(torch.float32) if torch.is_tensor(w) else w.astype(np.float32) for w in processed]
            S3Tokenizer._prepare_audio = _patched_prepare

            # 3. Patch VoiceEncoder (float64 avoids)
            from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder
            
            # Patch embeds_from_mels which was the source of the recent crash (calls .to(self.device) on float64)
            _orig_embeds_from_mels = VoiceEncoder.embeds_from_mels
            def _patched_embeds_from_mels(self_ve, mels, *args, **kwargs):
                if torch.is_tensor(mels):
                    mels = mels.to(torch.float32)
                elif isinstance(mels, list):
                    mels = [m.astype(np.float32) if isinstance(m, np.ndarray) else m for m in mels]
                return _orig_embeds_from_mels(self_ve, mels, *args, **kwargs)
            VoiceEncoder.embeds_from_mels = _patched_embeds_from_mels

            # Also patch inference just in case
            _orig_inference = VoiceEncoder.inference
            def _patched_inference(self_ve, mels, *args, **kwargs):
                return _orig_inference(self_ve, mels.to(torch.float32), *args, **kwargs)
            VoiceEncoder.inference = _patched_inference

            # 4. Patch AlignmentStreamAnalyzer (Fix SDPA/output_attentions error)
            try:
                from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
                _orig_add_spy = AlignmentStreamAnalyzer._add_attention_spy
                def _patched_add_spy(self_asa, tfmr, *args, **kwargs):
                    if hasattr(tfmr, 'config'):
                        # Force "eager" attention implementation if we want output_attentions
                        # This avoids the "SDPA does not support output_attentions" error in transformers
                        if getattr(tfmr.config, "_attn_implementation", None) == "sdpa":
                            logger.info("  Mapping tfmr.config.attn_implementation: sdpa -> eager")
                            tfmr.config._attn_implementation = "eager"
                    return _orig_add_spy(self_asa, tfmr, *args, **kwargs)
                AlignmentStreamAnalyzer._add_attention_spy = _patched_add_spy
            except ImportError:
                pass

            # 5. Patch ChatterboxTurboTTS.norm_loudness (Source of float64 promotion)
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                _orig_norm = ChatterboxTurboTTS.norm_loudness
                def _patched_norm(self_tts, wav, *args, **kwargs):
                    res = _orig_norm(self_tts, wav, *args, **kwargs)
                    if isinstance(res, np.ndarray):
                        return res.astype(np.float32)
                    return res
                ChatterboxTurboTTS.norm_loudness = _patched_norm
            except ImportError:
                pass # Already patched or not available

            logger.info("✓ Comprehensive MPS and CPU-compatibility patches applied")
        except Exception as e:
            logger.warning(f"Failed to apply stability patches: {e}")

    def _get_model(self, language_id: str | None = None):
        is_turbo = (language_id is None or language_id.lower() == "en")
        model_key = "turbo" if is_turbo else "multilingual"
        
        if model_key not in self.models:
            try:
                if is_turbo:
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    logger.info("Loading official ChatterboxTurboTTS (English)...")
                    model = ChatterboxTurboTTS.from_pretrained(self.device)
                else:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    logger.info("Loading official ChatterboxMultilingualTTS (23 Languages)...")
                    device_obj = torch.device(self.device)
                    model = ChatterboxMultilingualTTS.from_pretrained(device_obj)
                
                # Monkeypatch watermarker to be optional
                if hasattr(model, "watermarker"):
                    orig_apply = model.watermarker.apply_watermark
                    def patched_apply(wav, sample_rate):
                        if getattr(self, "use_watermark", True):
                            return orig_apply(wav, sample_rate)
                        return wav
                    model.watermarker.apply_watermark = patched_apply
                
                self.models[model_key] = model
            except Exception as e:
                logger.error(f"Failed to load {model_key} model: {e}")
                raise
        
        return self.models[model_key]

    def _auto_detect_device(self) -> str:
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
        language_id: str | None = None,
        watermark: bool = True
    ) -> torch.Tensor:
        """
        Generate speech using the appropriate official API (Turbo vs Multilingual).
        """
        model = self._get_model(language_id)
        self.use_watermark = watermark
        
        lang = language_id.lower() if language_id else "en"
        logger.info(f"Generating with {'Turbo' if lang == 'en' else 'Multilingual'} API. Language: {lang} (Watermark: {watermark})")
        
        if lang == "en":
            # Turbo generate signature
            return model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=0.8,
                top_k=1000,
                top_p=0.95,
                repetition_penalty=1.2,
                norm_loudness=True
            )
        else:
            # Multilingual generate signature
            return model.generate(
                text=text,
                language_id=lang,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=0.8,
                repetition_penalty=2.0,
                min_p=0.05,
                top_p=1.0
            )
    
    @property
    def sr(self) -> int:
        return 24000

def _synthesize_with_pytorch(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    multilingual: bool = False,
    model_type: str = "chatterbox-turbo",
    device: str | None = None,
    watermark: bool = True
):
    # Determine model key for caching
    model_key = f"{model_type}_{device}"
    
    global _MODEL_CACHE
    if model_key in _MODEL_CACHE:
        wrapper = _MODEL_CACHE[model_key]
    else:
        logger.info(f"Loading Chatterbox model: {model_key}...")
        wrapper = ChatterboxWrapper(device=device, model_type=model_type)
        _MODEL_CACHE[model_key] = wrapper

    # Determine reference audio
    ref_audio = source_wav
    if not ref_audio:
        ref_audio = VOICE_PRESETS.get(language, VOICE_PRESETS.get("en"))

    try:
        wav = wrapper.generate(
            text=text,
            audio_prompt_path=ref_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language_id=language,
            watermark=watermark
        )
    except AssertionError as e:
        if "Audio prompt must be longer than 5 seconds" in str(e):
            raise ValueError("Chatterbox-Turbo requires an audio prompt longer than 5 seconds for voice cloning.")
        raise e
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav
    if wav_np.ndim > 1:
        wav_np = wav_np.squeeze()
    sf.write(output_wav, wav_np, wrapper.sr)
    sf.write(output_wav, wav_np, wrapper.sr)
    logger.info(f"✓ Saved audio to {output_wav}")
    
    # Cleanup
    del wrapper
    if torch.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def _synthesize_with_mlx(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    model_type: str = "chatterbox-turbo",
    model_id: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    stream: bool = False
):
    try:
        from mlx_audio.tts.generate import generate_audio
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import ChatterboxTurboTTS as ChatterboxTurboModel
    except ImportError:
        try:
            from mlx_audio.tts.generate import generate_audio
            ChatterboxTurboModel = None
        except ImportError:
            raise ImportError(
                "MLX backend requires 'mlx-audio>=0.2.10' package. Install with:\n"
                "  uv pip install -U mlx-audio"
            )

    if ChatterboxTurboModel is not None:
        original_load_weights = ChatterboxTurboModel.load_weights
        def patched_load_weights(self, weights: list, strict: bool = False):
            original_load_weights(self, weights, strict=False)
        ChatterboxTurboModel.load_weights = patched_load_weights
    
    logger.info(f"Generating speech with MLX backend ({model_type})...")
    
    lang_code = language if language else "en"
    is_english = (lang_code == "en")
    
    # Determine reference audio
    ref_audio = source_wav
    if not ref_audio and voice and voice in VOICE_PRESETS:
        ref_audio = VOICE_PRESETS[voice]
    elif not ref_audio and voice and os.path.exists(str(voice)):
        ref_audio = voice
    if not ref_audio:
        ref_audio = VOICE_PRESETS.get(language, VOICE_PRESETS.get("en"))
    if not ref_audio:
        raise ValueError(f"Chatterbox (MLX) requires a reference audio file.")

    # Determine target model ID
    if model_id:
        target_model = model_id
    else:
        # Hybrid MLX logic: Turbo for English, Standard for Multilingual
        if is_english:
            target_model = "mlx-community/Chatterbox-Turbo-4bit"
        else:
            logger.info(f"Language '{lang_code}' is not English. Using Multilingual (4-bit) MLX backend.")
            target_model = "mlx-community/chatterbox-4bit" # Best 4-bit multilingual fallback
    
    output_dir = os.path.dirname(output_wav) or "."
    output_name = os.path.splitext(os.path.basename(output_wav))[0]
    file_prefix = os.path.join(output_dir, output_name)
    
    voice_map = {
        "en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "it": "if_sara",
        "pt": "pf_dora", "de": "af_sarah", "ru": "af_nicole", "tr": "af_river",
        "hi": "hf_alpha", "ja": "jf_alpha", "zh": "zf_xiaoxiao"
    }
    kokoro_voice = voice if voice else voice_map.get(language, "af_heart")
    
    kwargs = {
        "text": text,
        "model": target_model,
        "ref_audio": ref_audio,
        "ref_text": ".",
        "file_prefix": file_prefix,
        "lang_code": lang_code,
        "voice": kokoro_voice,
        "speed": speed,
        "stream": False,
        "play": stream,
        "verbose": True
    }
    
    generate_audio(**kwargs)
    
    expected_output = f"{file_prefix}_000.wav"
    if not os.path.exists(expected_output) and os.path.exists(f"{file_prefix}.wav"):
        expected_output = f"{file_prefix}.wav"
    
    if os.path.exists(expected_output):
        if os.path.abspath(expected_output) != os.path.abspath(output_wav):
            if os.path.exists(output_wav):
                os.remove(output_wav)
            os.rename(expected_output, output_wav)
    else:
        raise RuntimeError(f"MLX generation failed: {expected_output} not found")

def synthesize_with_chatterbox_turbo(
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
    stream: bool = False,
    device: str | None = None,
    watermark: bool = True
):
    if use_mlx:
         _synthesize_with_mlx(
             text=text, 
             output_wav=output_wav, 
             source_wav=source_wav, 
             exaggeration=exaggeration, 
             cfg_weight=cfg_weight, 
             language=language, 
             model_id=model_id,
             voice=voice,
             speed=speed,
             stream=stream
         )
    else:
        _synthesize_with_pytorch(
            text=text, 
            output_wav=output_wav, 
            source_wav=source_wav, 
            exaggeration=exaggeration, 
            cfg_weight=cfg_weight, 
            language=language, 
            multilingual=multilingual,
            device=device,
            watermark=watermark
        )
