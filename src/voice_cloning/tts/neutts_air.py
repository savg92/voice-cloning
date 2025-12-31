"""
NeuTTS Air Wrapper
Voice cloning TTS using NeuTTS Air (Neuphonic)
"""

import logging
import os
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


class NeuTTSAirTTS:
    """
    Wrapper for NeuTTS Air voice cloning TTS.
    Requires 3+ seconds of reference audio for voice cloning.
    """
    
    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-air-q4-gguf",
        backbone_device: str = "cpu",
        codec_repo: str = "neuphonic/neucodec",
        codec_device: str = "cpu"
    ):
        """
        Initialize NeuTTS Air TTS.
        
        Args:
            backbone_repo: Backbone model repository
            backbone_device: Device for backbone (cpu/cuda)
            codec_repo: Codec repository
            codec_device: Device for codec (cpu/cuda)
        """
        # Configure phonemizer to use espeak-ng
        import os
        print("DEBUG: Starting NeuTTSAirTTS initialization...")
        
        if 'PHONEMIZER_ESPEAK_PATH' not in os.environ:
            # Try to find espeak-ng
            espeak_ng_path = '/opt/homebrew/bin/espeak-ng'
            if os.path.exists(espeak_ng_path):
                os.environ['PHONEMIZER_ESPEAK_PATH'] = espeak_ng_path
                os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/lib/libespeak-ng.dylib'
                print(f"DEBUG: Configured phonemizer to use espeak-ng at {espeak_ng_path}")
                logger.info(f"Configured phonemizer to use espeak-ng at {espeak_ng_path}")
        
        print("DEBUG: Importing NeuTTSAir module...")
        try:
            # Add models directory to path
            import sys
            from pathlib import Path
            models_dir = Path(__file__).parent.parent.parent.parent / "models"
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            print(f"DEBUG: Added {models_dir} to sys.path")
            
            from neuttsair.neutts import NeuTTSAir
            self.NeuTTSAir = NeuTTSAir
            print("DEBUG: Successfully imported NeuTTSAir")
        except ImportError as e:
            print(f"DEBUG: Import failed: {e}")
            raise ImportError(
                f"neuttsair module not found. Please ensure the neuttsair directory "
                f"is in models/ folder. Error: {e}"
            )
        
        logger.info("Initializing NeuTTS Air...")
        logger.info(f"  Backbone: {backbone_repo} ({backbone_device})")
        logger.info(f"  Codec: {codec_repo} ({codec_device})")
        
        print("\n⏳ Loading NeuTTS Air (first run may download models - several GB)...")
        print(f"   Backbone: {backbone_repo}")
        print("   This may take 5-10 minutes on first run...")
        print("DEBUG: About to create NeuTTSAir instance...\n")
        
        self.tts = self.NeuTTSAir(
            backbone_repo=backbone_repo,
            backbone_device=backbone_device,
            codec_repo=codec_repo,
            codec_device=codec_device
        )
        
        print("DEBUG: NeuTTSAir instance created successfully!")
        logger.info("✓ NeuTTS Air initialized successfully")
        print("✓ Model loaded successfully!\n")
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        ref_audio_path: str,
        ref_text_path: str | None = None,
    ) -> str:
        """
        Synthesize speech with voice cloning.
        
        Args:
            text: Input text to synthesize
            output_path: Path to save output WAV
            ref_audio_path: Path to reference audio (3+ seconds recommended)
            ref_text_path: Path to reference text file (transcript of ref audio)
                          OR the literal transcript text itself.
                          If None/empty, will look for .txt file with same name as ref_audio
            
        Returns:
            Path to output file
        """
        print("DEBUG: Starting synthesis...")
        
        ref_text = None
        
        # 1. Handle None or empty string - trigger auto-detection
        if not ref_text_path:
            # Try to find matching .txt file
            ref_audio_pathlib = Path(ref_audio_path)
            detected_path = ref_audio_pathlib.with_suffix('.txt')
            print(f"DEBUG: No reference text provided, checking for auto-detection: {detected_path}")
            
            if detected_path.exists():
                print(f"DEBUG: Auto-detected ref_text_path: {detected_path}")
                with open(detected_path) as f:
                    ref_text = f.read().strip()
            else:
                print(f"DEBUG: Auto-detection failed, no .txt file found at: {detected_path}")
                raise FileNotFoundError(
                    f"Reference text not provided and no matching .txt file found at {detected_path}.\n"
                    f"Please provide a transcript of the reference audio."
                )
        
        # 2. Check if the provided string is an existing file path
        elif os.path.exists(ref_text_path):
            print(f"DEBUG: Loading reference text from file: {ref_text_path}")
            with open(ref_text_path) as f:
                ref_text = f.read().strip()
        
        # 3. Otherwise, treat it as the literal transcript text
        else:
            print(f"DEBUG: Using provided string as literal reference text")
            ref_text = ref_text_path.strip()
        
        logger.info(f"Reference audio: {ref_audio_path}")
        logger.info(f"Reference text: {ref_text[:50]}...")
        
        # Encode reference
        print("DEBUG: Encoding reference audio...")
        ref_codes = self.tts.encode_reference(ref_audio_path)
        print("DEBUG: Reference encoded successfully")
        
        # Generate speech
        logger.info(f"Generating speech: {text[:50]}...")
        print(f"DEBUG: Generating speech with text: '{text[:60]}...'")
        wav = self.tts.infer(text, ref_codes, ref_text)
        print("DEBUG: Speech generated successfully")
        
        # Save
        print(f"DEBUG: Saving audio to: {output_path}")
        sf.write(output_path, wav, 24000)
        logger.info(f"✓ Audio saved to: {output_path}")
        print("DEBUG: Audio saved, synthesis complete!")
        
        return output_path


def synthesize_with_neutts_air(
    text: str,
    output_path: str,
    ref_audio: str,
    ref_text: str | None = None,
    backbone: str = "neuphonic/neutts-air-q4-gguf",
    device: str = "cpu"
) -> str:
    """
    Convenience function to synthesize speech with NeuTTS Air.
    
    Args:
        text: Text to synthesize
        output_path: Output WAV file path
        ref_audio: Reference audio file (3+ seconds)
        ref_text: Reference text file (transcript)
        backbone: Backbone model repo
        device: Device (cpu/cuda)
        
    Returns:
        Path to output file
    """
    tts = NeuTTSAirTTS(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo="neuphonic/neucodec",
        codec_device=device
    )
    
    return tts.synthesize(
        text=text,
        output_path=output_path,
        ref_audio_path=ref_audio,
        ref_text_path=ref_text
    )
