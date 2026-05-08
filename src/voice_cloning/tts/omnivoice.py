"""
This module provides an interface to interact with the k2-fsa OmniVoice TTS model.
Supports voice cloning, voice design, and massively multilingual synthesis.
"""

import logging
from pathlib import Path
from typing import Optional

import os
import platform
import soundfile as sf
import torch
import torchaudio
import numpy as np

from .utils import get_best_device

logger = logging.getLogger(__name__)

class OmniVoiceTTS:
    """
    Interface for k2-fsa OmniVoice TTS model.
    Supports voice cloning, voice design, and massively multilingual synthesis.
    """

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", device: Optional[str] = None):
        """
        Initializes the OmniVoice model.

        Args:
            model_id (str): Hugging Face model ID.
            device (Optional[str]): Device to run the model on (cuda, mps, cpu).
        """
        if platform.system() == "Darwin":
            # Help torchcodec find Homebrew's FFmpeg libraries if needed
            hb_lib_path = "/opt/homebrew/lib"
            if os.path.exists(hb_lib_path):
                import ctypes
                try:
                    # Pre-load FFmpeg libraries globally so torchcodec can find the symbols
                    # Order matters for dependencies
                    for lib in ["libavutil", "libswresample", "libavcodec", "libavformat", "libswscale"]:
                        lib_file = f"{hb_lib_path}/{lib}.dylib"
                        if os.path.exists(lib_file):
                            ctypes.CDLL(lib_file, mode=ctypes.RTLD_GLOBAL)
                    logger.info("Pre-loaded FFmpeg libraries from Homebrew for torchcodec compatibility.")
                except Exception as e:
                    logger.warning(f"Failed to pre-load FFmpeg libraries: {e}")

        if device is None:
            device = get_best_device()
            logger.info(f"Auto-selected best device for OmniVoice: {device}")
        
        self.device = device
        logger.info(f"Loading OmniVoice model '{model_id}' on {device}...")
        
        try:
            # Prepare initialization arguments
            init_kwargs = {}
            if device == "mps":
                # Eager attention is more stable on MPS for some models
                init_kwargs["attn_implementation"] = "eager"
                logger.info("Using attn_implementation='eager' for MPS stability.")

            from omnivoice import OmniVoice
            self.model = OmniVoice.from_pretrained(model_id, **init_kwargs)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load OmniVoice model: {e}")
            raise

        # OmniVoice typically uses 24kHz sampling rate
        self.sampling_rate = 24000 

    def synthesize(
        self,
        text: str,
        output_path: str = "omnivoice_output.wav",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> str:
        """
        Synthesizes speech using OmniVoice.

        Args:
            text (str): Text to synthesize.
            output_path (str): Path to save the generated audio.
            ref_audio (Optional[str]): Path to reference audio for voice cloning.
            ref_text (Optional[str]): Transcript of the reference audio.
            instruct (Optional[str]): Natural language instruction for voice design.
            language (Optional[str]): Language code.
            speed (float): Speech speed multiplier.
            **kwargs: Additional generation parameters.

        Returns:
            str: Path to the generated audio file.
        """
        logger.info(f"Synthesizing with OmniVoice: '{text[:50]}...' (Cloning: {ref_audio is not None}, Design: {instruct is not None})")
        
        # Prepare arguments
        gen_kwargs = {
            "text": text,
            "speed": speed,
            **kwargs
        }
        
        if ref_audio:
            ref_audio_path = Path(ref_audio)
            if not ref_audio_path.exists():
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
            
            # Pass path directly to OmniVoice to see if it handles memory better
            gen_kwargs["ref_audio"] = str(ref_audio_path)
            logger.info(f"Using reference audio path: {ref_audio_path}")

        if ref_text:
            gen_kwargs["ref_text"] = ref_text
        
        if instruct:
            gen_kwargs["instruct"] = instruct
        if language:
            gen_kwargs["language"] = language

        try:
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            
            if not outputs or len(outputs) == 0:
                raise RuntimeError("OmniVoice failed to generate audio.")
            
            # OmniVoice returns a list of numpy arrays (one for each input text)
            audio = outputs[0]
            
            # Ensure output directory exists
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio
            sf.write(str(output_path), audio, self.sampling_rate)
            logger.info(f"✓ OmniVoice synthesis complete: {output_path}")
            
            return str(output_path)
        except Exception as e:
            logger.error(f"OmniVoice synthesis failed: {e}")
            raise RuntimeError(f"OmniVoice generation failed: {e}")

def synthesize_speech(
    text: str,
    output_path: str = "omnivoice_output.wav",
    reference: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    language: Optional[str] = None,
    speed: float = 1.0,
    device: Optional[str] = None,
    **kwargs
) -> str:
    """
    Functional interface for OmniVoice synthesis.
    Improved for memory safety on 8GB devices by transcribing reference audio
    BEFORE loading the main model.
    """
    # 1. Handle auto-transcription first to save memory
    if reference and not ref_text:
        logger.info("Reference text missing. Transcribing with light model (faster-whisper tiny) BEFORE loading OmniVoice...")
        try:
            from faster_whisper import WhisperModel
            # Use tiny model on CPU for minimal memory footprint
            model = WhisperModel("tiny", device="cpu", compute_type="float32")
            segments, _ = model.transcribe(str(reference), beam_size=1)
            ref_text = " ".join([s.text for s in segments]).strip()
            logger.info(f"Auto-transcribed reference: '{ref_text}'")
            # Immediate cleanup
            del model
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as asr_e:
            logger.warning(f"Failed to auto-transcribe with faster-whisper: {asr_e}")

    # 2. Now load the model and synthesize
    tts = OmniVoiceTTS(device=device)
    return tts.synthesize(
        text=text,
        output_path=output_path,
        ref_audio=reference,
        ref_text=ref_text,
        instruct=instruct,
        language=language,
        speed=speed,
        **kwargs
    )
