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
import logging
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class ChatterboxWrapper:
    """
    Wrapper for Chatterbox TTS models.
    
    Supports both English-only and Multilingual variants.
    """
    
    def __init__(self, device: Optional[str] = None, multilingual: bool = False):
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
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language_id: Optional[str] = None
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
    source_wav: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: Optional[str] = None,
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
    source_wav: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: Optional[str] = None
):
    """
    Synthesize speech using MLX backend (Apple Silicon optimized, 4-bit quantized).
    """
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError:
        raise ImportError(
            "MLX backend requires 'mlx-audio-plus' package. Install with:\n"
            "  uv pip install -U mlx-audio-plus\n"
            "Or use use_mlx=False to use PyTorch backend."
        )
    
    logger.info("Generating speech with MLX backend (Chatterbox-TTS-4bit)...")
    logger.info(f"Exaggeration={exaggeration}, CFG Weight={cfg_weight}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_prefix = os.path.join(tmpdir, "chatterbox_mlx")
        
        # MLX Chatterbox requires ref_audio and ref_text for voice cloning
        # If no reference is provided, we'll use default voice by passing None
        if source_wav:
            ref_audio = source_wav
            logger.info(f"Using reference audio for voice cloning: {source_wav}")
        else:
            # Try to find a default reference in samples
            default_ref = "samples/anger.wav"
            if os.path.exists(default_ref):
                ref_audio = default_ref
                logger.warning(f"No reference audio provided. Using default for testing: {default_ref}")
                logger.warning("To clone a specific voice, provide --reference path/to/audio.wav")
            else:
                 raise ValueError(
                     "Chatterbox (MLX) requires a reference audio file for voice cloning.\n"
                     "Please provide one using --reference path/to/audio.wav"
                 )
        
        # Reference text is optional for zero-shot in some implementations, 
        # but mlx-audio might require it or handle it. Passing "." as placeholder if None.
        ref_text = "."
        
        try:
            # Pass all control parameters to MLX backend via kwargs
            # Map 'language' to 'lang_code' if provided (default 'a' for American English)
            lang_code = language if language else "a"
            
            generate_audio(
                text=text,
                model="mlx-community/Chatterbox-TTS-4bit",
                ref_audio=ref_audio,
                ref_text=ref_text,
                file_prefix=file_prefix,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                lang_code=lang_code
            )
            
            # MLX generates file_prefix + ".wav" (or similar depending on version)
            # Latest mlx-audio-plus might use output_path directly or append ext
            
            # Check for potential output files
            possible_files = [
                f"{file_prefix}.wav",
                f"{file_prefix}_0.wav",
                f"{file_prefix}_000.wav"
            ]
            
            # Also check if it just wrote to output_path if it ends in .wav (but we passed directory-like prefix)
            
            generated_file = None
            for f in possible_files:
                if os.path.exists(f):
                    generated_file = f
                    break
            
            if not generated_file:
                # Fallback: list dir to find any wav
                files = os.listdir(tmpdir)
                wavs = [f for f in files if f.endswith(".wav")]
                if wavs:
                    generated_file = os.path.join(tmpdir, wavs[0])
            
            if not generated_file:
                raise RuntimeError(f"MLX did not generate expected output file. Found: {os.listdir(tmpdir)}")
            
            # Move to final output
            import shutil
            shutil.move(generated_file, output_wav)
            
            logger.info(f"✓ MLX synthesis complete: {output_wav}")
            
        except Exception as e:
            logger.error(f"MLX synthesis failed: {e}")
            raise RuntimeError(f"MLX generation failed: {e}")


def synthesize_with_chatterbox(
    text: str,
    output_wav: str,
    source_wav: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: Optional[str] = None,
    multilingual: bool = False,
    use_mlx: bool = False
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
    
    Supported languages (multilingual model, PyTorch only):
        ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, 
        pl, pt, ru, sv, sw, tr, zh
    """
    # Auto-enable multilingual if language is specified and not English
    if language and language != "en":
        multilingual = True
    
    # MLX backend
    if use_mlx:
         _synthesize_with_mlx(text, output_wav, source_wav, exaggeration, cfg_weight, language)
    else:
        _synthesize_with_pytorch(text, output_wav, source_wav, exaggeration, cfg_weight, language, multilingual)

