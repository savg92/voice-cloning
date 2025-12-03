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
    cfg_weight: float = 0.5
):
    """
    Synthesize speech using MLX backend (Apple Silicon optimized, 4-bit quantized).
    
    Note: MLX Chatterbox is English-only and does not support multilingual synthesis.
    """
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError:
        raise ImportError(
            "MLX backend requires 'mlx-audio' package. Install with:\n"
            "  pip install mlx-audio\n"
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
            ref_text = "."  # Placeholder text for voice cloning
            logger.info(f"Using reference audio for voice cloning: {source_wav}")
        else:
            ref_audio = None
            ref_text = None
            logger.info("Using default voice (no reference audio)")
        
        try:
            # Note: MLX audio doesn't expose exaggeration/cfg_weight in the same way
            # The generate_audio function has different parameters
            generate_audio(
                text=text,
                model_path="mlx-community/Chatterbox-TTS-4bit",
                ref_audio=ref_audio,
                ref_text=ref_text,
                file_prefix=file_prefix,
            )
            
            # MLX generates file_prefix_000.wav
            generated_file = f"{file_prefix}_000.wav"
            if not os.path.exists(generated_file):
                # Try without sequence for older versions
                generated_file = f"{file_prefix}.wav"
                if not os.path.exists(generated_file):
                    raise RuntimeError("MLX did not generate expected output file")
            
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
    
    # MLX backend doesn't support multilingual
    if use_mlx and multilingual:
        logger.warning("MLX backend is English-only. Falling back to PyTorch for multilingual synthesis.")
        use_mlx = False
    
    if use_mlx:
        _synthesize_with_mlx(text, output_wav, source_wav, exaggeration, cfg_weight)
    else:
        _synthesize_with_pytorch(text, output_wav, source_wav, exaggeration, cfg_weight, language, multilingual)

