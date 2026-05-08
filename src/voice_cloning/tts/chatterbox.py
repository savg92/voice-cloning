
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
    
    def __init__(self, device: str | None = None, model_type: str = "chatterbox", multilingual: bool = False):
        """
        Initialize Chatterbox TTS.
        
        Args:
            device: Device to use ('cuda', 'cpu', 'mps'). Auto-detected if None.
            model_type: 'chatterbox'
            multilingual: If True, load multilingual model. Default is English-only.
        """
        self.device = device or self._auto_detect_device()
        self.multilingual = multilingual
        self.model_type = model_type
        
        repo_id = "ResembleAI/chatterbox"
        
        logger.info(f"Initializing {model_type} ({'Multilingual' if multilingual else 'English'}) on {self.device}")
        
        try:
            # Fix for torch.load issues on non-CUDA devices (library uses hardcoded CUDA loads)
            import torch
            _orig_load = torch.load
            def _patched_load(f, *args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = 'cpu'
                return _orig_load(f, *args, **kwargs)

            # Fix for transformers 4.49+ requiring 'eager' for output_attentions
            from transformers import PretrainedConfig
            _orig_init = PretrainedConfig.__init__
            def _patched_config_init(self, *args, **kwargs):
                if 'output_attentions' in kwargs and kwargs['output_attentions']:
                    kwargs['attn_implementation'] = 'eager'
                # Also force eager if we know it is a Llama-based model in Chatterbox
                if hasattr(self, 'model_type') and self.model_type == 'llama':
                     kwargs['attn_implementation'] = 'eager'
                _orig_init(self, *args, **kwargs)
            PretrainedConfig.__init__ = _patched_config_init

            if multilingual:
                import chatterbox.mtl_tts
                chatterbox.mtl_tts.torch.load = _patched_load
                chatterbox.mtl_tts.REPO_ID = repo_id
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                import chatterbox.tts
                chatterbox.tts.torch.load = _patched_load
                chatterbox.tts.REPO_ID = repo_id
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                
            logger.info(f"✓ {model_type} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        from .utils import get_best_device
        return get_best_device()
    
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
        """
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight
        }
        
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = audio_prompt_path
        
        # Multilingual model requires language_id
        if self.multilingual:
            kwargs["language_id"] = language_id or "en"
        
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
    multilingual: bool = False,
    model_type: str = "chatterbox",
    device: str | None = None
):
    """
    Synthesize speech using PyTorch backend (chatterbox-tts library).
    """
    wrapper = ChatterboxWrapper(device=device, model_type=model_type, multilingual=multilingual)
    
    wav = wrapper.generate(
        text,
        audio_prompt_path=source_wav,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        language_id=language
    )
    
    # Save audio
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav
        
    if wav_np.ndim > 1:
        wav_np = wav_np.squeeze()
        
    sf.write(output_wav, wav_np, wrapper.sr)
    logger.info(f"✓ Saved audio to {output_wav}")


def _synthesize_with_mlx(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    model_type: str = "chatterbox",
    model_id: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    stream: bool = False
):
    """
    Synthesize speech using MLX backend.
    """
    try:
        from mlx_audio.tts.generate import generate_audio
        from mlx_audio.tts.models.chatterbox.chatterbox import Model as ChatterboxModel
    except ImportError:
        try:
            from mlx_audio.tts.generate import generate_audio
            ChatterboxModel = None
        except ImportError:
            raise ImportError(
                "MLX backend requires 'mlx-audio>=0.2.10' package. Install with:\n"
                "  uv pip install -U mlx-audio"
            )

    # Apply leniency patch for weight loading
    if ChatterboxModel is not None:
        original_load_weights = ChatterboxModel.load_weights
        def patched_load_weights(self, weights: list, strict: bool = False):
            original_load_weights(self, weights, strict=False)
        ChatterboxModel.load_weights = patched_load_weights
    
    logger.info(f"Generating speech with MLX backend ({model_type})...")
    
    # Determine reference audio
    ref_audio = source_wav
    if not ref_audio and voice and voice in VOICE_PRESETS:
        ref_audio = VOICE_PRESETS[voice]
    elif not ref_audio and voice and os.path.exists(str(voice)):
        ref_audio = voice
    
    if not ref_audio:
        ref_audio = VOICE_PRESETS.get(language, VOICE_PRESETS.get("en"))
        
    if not ref_audio:
        raise ValueError(f"Chatterbox (MLX) requires a reference audio file. None found for voice={{voice}}, lang={{language}}")

    # Determine target model ID
    target_model = model_id or "mlx-community/chatterbox-4bit"

    # Map language code
    lang_code = map_lang_code(language) if language else "a"
    
    # Prepare output paths
    output_dir = os.path.dirname(output_wav) or "."
    output_name = os.path.splitext(os.path.basename(output_wav))[0]
    file_prefix = os.path.join(output_dir, output_name)
    
    # Voice mapping
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
    
    # Handle mlx-audio's _000.wav suffix
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
    stream: bool = False,
    device: str | None = None
):
    """
    Synthesize speech from text using Chatterbox TTS.
    """
    # Auto-enable multilingual if language is specified and not English
    if language and language != "en":
        multilingual = True
    
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
            device=device
        )
