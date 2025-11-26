"""
Chatterbox TTS Wrapper

Supports both English (ChatterboxTTS) and Multilingual (ChatterboxMultilingualTTS)
with voice cloning and exaggeration control.

Features:
- Zero-shot voice cloning
- Exaggeration/intensity control (0-1, default 0.5)
- CFG weight for generation control (0-1, default 0.5)
- Multilingual support (23 languages)
"""

import torch
import torchaudio as ta
import logging
from pathlib import Path
from typing import Optional

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


def synthesize_with_chatterbox(
    text: str,
    output_wav: str,
    source_wav: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: Optional[str] = None,
    multilingual: bool = False
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
        multilingual: Use multilingual model (supports 23 languages)
    
    Supported languages (multilingual model):
        ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, 
        pl, pt, ru, sv, sw, tr, zh
    """
    # Auto-enable multilingual if language is specified and not English
    if language and language != "en":
        multilingual = True
    
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
