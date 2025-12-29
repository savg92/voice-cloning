"""
Tests for NeuTTS Air TTS implementation
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.neutts_air import NeuTTSAirTTS, synthesize_with_neutts_air


# Skip tests if neuttsair module not available
try:
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    from neuttsair.neutts import NeuTTSAir  # noqa: F401
    NEUTTS_AVAILABLE = True
except ImportError:
    NEUTTS_AVAILABLE = False

SKIP_REASON = "neuttsair module not available"
pytestmark = pytest.mark.skipif(not NEUTTS_AVAILABLE, reason=SKIP_REASON)


# Sample files
SAMPLES_DIR = Path("samples/neutts_air")
REF_AUDIO = SAMPLES_DIR / "dave.wav"
REF_TEXT = SAMPLES_DIR / "dave.txt"


class TestNeuTTSAir:
    """Test suite for NeuTTS Air TTS."""
    
    @pytest.fixture
    def temp_output(self):
        """Create a temporary output file."""
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.mark.skipif(not REF_AUDIO.exists(), reason="Sample audio not found")
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        tts = NeuTTSAirTTS()
        assert tts is not None
        assert tts.tts is not None
    
    @pytest.mark.skipif(not REF_AUDIO.exists() or not REF_TEXT.exists(), 
                        reason="Sample files not found")
    def test_basic_voice_cloning(self, temp_output):
        """Test basic voice cloning synthesis."""
        tts = NeuTTSAirTTS()
        
        text = "Hello, this is a test of voice cloning."
        result = tts.synthesize(
            text=text,
            output_path=temp_output,
            ref_audio_path=str(REF_AUDIO),
            ref_text_path=str(REF_TEXT)
        )
        
        assert result == temp_output
        assert os.path.exists(temp_output)
        
        # Verify audio
        audio, sr = sf.read(temp_output)
        assert sr == 24000  # NeuTTS Air uses 24kHz
        assert len(audio) > 0
    
    @pytest.mark.skipif(not REF_AUDIO.exists(), reason="Sample audio not found")
    def test_auto_detect_ref_text(self, temp_output):
        """Test auto-detection of reference text file."""
        tts = NeuTTSAirTTS()
        
        text = "Testing automatic reference text detection."
        result = tts.synthesize(
            text=text,
            output_path=temp_output,
            ref_audio_path=str(REF_AUDIO),
            ref_text_path=None  # Auto-detect
        )
        
        assert os.path.exists(result)
    
    @pytest.mark.skipif(not REF_AUDIO.exists() or not REF_TEXT.exists(),
                        reason="Sample files not found")
    def test_convenience_function(self, temp_output):
        """Test the convenience wrapper function."""
        result = synthesize_with_neutts_air(
            text="Testing the convenience function.",
            output_path=temp_output,
            ref_audio=str(REF_AUDIO),
            ref_text=str(REF_TEXT)
        )
        
        assert result == temp_output
        assert os.path.exists(temp_output)
    
    @pytest.mark.skipif(not REF_AUDIO.exists() or not REF_TEXT.exists(),
                        reason="Sample files not found")
    def test_longer_text(self, temp_output):
        """Test synthesis with longer text."""
        tts = NeuTTSAirTTS()
        
        text = (
            "This is a longer piece of text to test the voice cloning system. "
            "It should maintain the cloned voice characteristics throughout "
            "the entire generated speech, producing natural and  realistic output."
        )
        
        result = tts.synthesize(
            text=text,
            output_path=temp_output,
            ref_audio_path=str(REF_AUDIO),
            ref_text_path=str(REF_TEXT)
        )
        
        assert os.path.exists(result)
        audio, sr = sf.read(result)
        # Longer text should produce longer audio
        assert len(audio) > 24000  # At least 1 second
    
    def test_missing_ref_text_raises_error(self, temp_output):
        """Test that missing reference text raises appropriate error."""
        tts = NeuTTSAirTTS()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            fake_audio = f.name
        
        try:
            with pytest.raises(FileNotFoundError):
                tts.synthesize(
                    text="Test",
                    output_path=temp_output,
                    ref_audio_path=fake_audio,
                    ref_text_path=None
                )
        finally:
            if os.path.exists(fake_audio):
                os.remove(fake_audio)


def test_sample_files_exist():
    """Test that sample reference files exist."""
    if not NEUTTS_AVAILABLE:
        pytest.skip(SKIP_REASON)
    
    assert SAMPLES_DIR.exists(), f"Samples directory not found: {SAMPLES_DIR}"
    assert REF_AUDIO.exists(), f"Reference audio not found: {REF_AUDIO}"
    assert REF_TEXT.exists(), f"Reference text not found: {REF_TEXT}"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
