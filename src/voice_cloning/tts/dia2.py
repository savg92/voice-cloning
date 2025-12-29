"""
Dia2-1B TTS Model Wrapper

This module provides a wrapper for the Dia2-1B text-to-speech model from Nari Labs.
Uses the official dia2 library from nari-labs/dia2 repository.

Model: https://huggingface.co/nari-labs/Dia2-1B
GitHub: https://github.com/nari-labs/dia2
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Dia2TTS:
    """
    Wrapper for Dia2-1B TTS model.
    
    Dia2 is a streaming dialogue TTS model capable of generating ultra-realistic
    conversational audio in real-time.
    
    Args:
        model_name: HuggingFace model name (default: "nari-labs/Dia2-1B")
        device: Device to run model on ("cuda", "mps", "cpu", or None for auto)
        dtype: Model precision ("float32", "float16", "bfloat16")
    """
    
    def __init__(
        self,
        model_name: str = "nari-labs/Dia2-1B",
        device: Optional[str] = None,
        dtype: str = "bfloat16"
    ):
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Initializing Dia2 model: {model_name}")
        logger.info(f"Device: {device}, dtype: {dtype}")
        
        try:
            from dia2 import Dia2, GenerationConfig, SamplingConfig
            
            self.Dia2 = Dia2
            self.GenerationConfig = GenerationConfig
            self.SamplingConfig = SamplingConfig
            
            # Load model from HuggingFace
            self.model = Dia2.from_repo(
                model_name,
                device=device,
                dtype=dtype
            )
            
            logger.info("✓ Dia2 model initialized successfully")
            
        except ImportError as e:
            logger.error(
                "dia2 library not found. Please install it:\n"
                "  git clone https://github.com/nari-labs/dia2.git\n"
                "  cd dia2\n"
                "  uv pip install -e .\n"
                "  uv pip install sphn whisper-timestamped"
            )
            raise ImportError(
                "dia2 library is required for Dia2-1B. "
                "See installation instructions above."
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Dia2 model: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        cfg_scale: float = 2.0,
        temperature: float = 0.8,
        top_k: int = 50,
        use_cuda_graph: bool = True,
        verbose: bool = False,
        prefix_speaker_1: Optional[str] = None,
        prefix_speaker_2: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text using Dia2.
        
        Args:
            text: Input text with speaker tags ([S1], [S2])
                  Example: "[S1] Hello! [S2] How are you?"
            output_path: Optional path to save output WAV file
            cfg_scale: Classifier-free guidance scale (default: 2.0)
            temperature: Sampling temperature (default: 0.8)
            top_k: Top-k sampling parameter (default: 50)
            use_cuda_graph: Enable CUDA graph optimization (default: True)
            verbose: Print generation progress (default: False)
            prefix_speaker_1: Path to audio file for voice cloning (speaker 1)
            prefix_speaker_2: Path to audio file for voice cloning (speaker 2)
            
        Returns:
            Audio waveform as numpy array (shape: [samples])
        """
        logger.info(f"Synthesizing text: {text[:100]}...")
        
        try:
            # Create generation config
            config = self.GenerationConfig(
                cfg_scale=cfg_scale,
                audio=self.SamplingConfig(temperature=temperature, top_k=top_k),
                use_cuda_graph=use_cuda_graph,
            )
            
            # Generate audio
            kwargs = {
                "config": config,
                "verbose": verbose,
            }
            
            # Add output path if specified
            if output_path:
                kwargs["output_wav"] = output_path
                
            # Add prefix audio for voice cloning if specified
            if prefix_speaker_1:
                kwargs["prefix_speaker_1"] = prefix_speaker_1
            if prefix_speaker_2:
                kwargs["prefix_speaker_2"] = prefix_speaker_2
                
            result = self.model.generate(text, **kwargs)
            
            # Extract waveform from result
            waveform = result.waveform
            
            # Convert torch tensor to numpy array if needed
            if hasattr(waveform, 'cpu') and hasattr(waveform, 'numpy'):
                # It's a torch tensor
                waveform = waveform.cpu().numpy()
            elif not isinstance(waveform, np.ndarray):
                # Convert other types to numpy
                waveform = np.array(waveform)
                
            # Ensure 1D array (mono)
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
                
            logger.info(f"✓ Generated {len(waveform)} samples")
            
            return waveform
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def __repr__(self):
        return (
            f"Dia2TTS(model_name='{self.model_name}', "
            f"device='{self.device}', dtype='{self.dtype}')"
        )
