"""
NVIDIA Canary-1B-v2 ASR Model Wrapper
A powerful 1-billion parameter model for speech transcription and translation across 25 European languages.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CanaryASR:
    """
    NVIDIA Canary-1B-v2 ASR Model Wrapper
    
    Supports:
    - Speech Transcription (ASR) for 25 languages
    - Speech Translation (AST) from English → 24 languages  
    - Speech Translation (AST) from 24 languages → English
    """
    
    def __init__(self):
        """Initialize the Canary ASR model."""
        self.model = None
        self._mode = "asr"  # Default mode
        self.supported_languages = [
            'bg', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'de', 
            'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ro', 'sk', 
            'sl', 'es', 'sv', 'ru', 'uk'
        ]
        
    def load_model(self) -> bool:
        """
        Load the Canary-1B-v2 model from NVIDIA.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Import NeMo components
            from nemo.collections.asr.models import ASRModel
            
            logger.info("Loading NVIDIA Canary-1B-v2 model...")
            self.model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
            logger.info("✓ Canary-1B-v2 model loaded successfully!")
            return True
            
        except ImportError as e:
            missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
            logger.error(f"Missing dependency: {missing_module}")
            
            if "nemo" in str(e).lower():
                logger.error("NeMo toolkit not properly installed.")
                logger.error("Try: uv add 'nemo-toolkit[asr]'")
            elif "pyannote" in str(e).lower():
                logger.error("PyAnnote not installed.")
                logger.error("Try: uv add pyannote.audio pyannote.core")
            elif "editdistance" in str(e).lower():
                logger.error("editdistance not installed.")
                logger.error("Try: uv add editdistance")
            else:
                logger.error(f"Missing dependency: {missing_module}")
                logger.error("Try installing with: uv add <package_name>")
            
            logger.error(f"Full error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Canary model: {e}")
            return False
    
    def transcribe(
        self, 
        audio_path: Union[str, Path],
        source_lang: str = "en",
        target_lang: str = "en",
        timestamps: bool = False,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Transcribe or translate audio using Canary-1B-v2.
        
        Args:
            audio_path: Path to audio file (.wav or .flac)
            source_lang: Source language code (e.g., 'en', 'fr', 'de')
            target_lang: Target language code for translation
            timestamps: Whether to include word/segment timestamps
            batch_size: Batch size for processing
            
        Returns:
            Dict containing transcribed/translated text and optional timestamps
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load Canary model")
        
        # Validate languages
        if source_lang not in self.supported_languages:
            raise ValueError(f"Source language '{source_lang}' not supported. Supported: {self.supported_languages}")
        if target_lang not in self.supported_languages:
            raise ValueError(f"Target language '{target_lang}' not supported. Supported: {self.supported_languages}")
        
        # Validate audio file
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in ['.wav', '.flac']:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}. Use .wav or .flac")
        
        try:
            logger.info(f"Processing audio: {audio_path}")
            logger.info(f"Source language: {source_lang}, Target language: {target_lang}")
            
            # Perform transcription/translation
            output = self.model.transcribe(
                audio=[str(audio_path)], 
                source_lang=source_lang, 
                target_lang=target_lang,
                timestamps=timestamps,
                batch_size=batch_size
            )
            
            result = {
                'text': output[0].text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'audio_path': str(audio_path)
            }
            
            # Add timestamps if requested
            if timestamps:
                result['timestamps'] = {
                    'word': output[0].timestamp.get('word', []) if hasattr(output[0], 'timestamp') else [],
                    'segment': output[0].timestamp.get('segment', []) if hasattr(output[0], 'timestamp') else []
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_file(
        self, 
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        source_lang: str = "en",
        target_lang: str = "en",
        timestamps: bool = False
    ) -> str:
        """
        Transcribe audio file and save result to text file.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output text file (optional)
            source_lang: Source language code
            target_lang: Target language code
            timestamps: Whether to include timestamps
            
        Returns:
            str: Path to output text file
        """
        audio_path = Path(audio_path)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = audio_path.with_suffix('.txt')
        else:
            output_path = Path(output_path)
            if output_path.suffix != '.txt':
                output_path = output_path.with_suffix('.txt')
        
        # Perform transcription
        result = self.transcribe(
            audio_path=audio_path,
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=timestamps
        )
        
        # Format output text
        output_text = result['text']
        
        if timestamps and 'timestamps' in result:
            output_text += "\n\n--- Timestamps ---\n"
            for segment in result['timestamps']['segment']:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('segment', '')
                output_text += f"{start:.2f}s - {end:.2f}s : {text}\n"
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        logger.info(f"✓ Transcription saved to: {output_path}")
        return str(output_path)


def get_canary() -> CanaryASR:
    """Get a Canary ASR instance."""
    return CanaryASR()


def transcribe_to_file(
    audio_path: Union[str, Path], 
    output_path: Union[str, Path],
    source_lang: str = "en",
    target_lang: str = "en"
) -> str:
    """
    Convenience function to transcribe audio file to text file.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for output file
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'en')
        
    Returns:
        str: Path to output text file
    """
    canary = get_canary()
    
    # Convert .wav output to .txt for transcription
    output_path = Path(output_path)
    if output_path.suffix == '.wav':
        output_path = output_path.with_suffix('.txt')
    
    return canary.transcribe_file(
        audio_path=audio_path,
        output_path=output_path,
        source_lang=source_lang,
        target_lang=target_lang,
        timestamps=True  # Include timestamps by default
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python canary.py <audio_file> [output_file] [source_lang] [target_lang]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    source_lang = sys.argv[3] if len(sys.argv) > 3 else "en"
    target_lang = sys.argv[4] if len(sys.argv) > 4 else "en"
    
    try:
        result_path = transcribe_to_file(audio_file, output_file, source_lang, target_lang)
        print(f"✓ Transcription completed: {result_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
