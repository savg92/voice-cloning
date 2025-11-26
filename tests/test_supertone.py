"""
Tests for Supertone TTS implementation
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voice_cloning.tts.supertone import SupertoneTTS, synthesize_with_supertone


# Skip all tests if models not available
MODELS_DIR = Path("models/supertonic")
SKIP_REASON = "Supertone models not downloaded. Run: git clone https://huggingface.co/Supertone/supertonic models/supertonic"
pytestmark = pytest.mark.skipif(not MODELS_DIR.exists(), reason=SKIP_REASON)


class TestSupertoneTTS:
    """Test suite for Supertone TTS."""
    
    @pytest.fixture
    def tts_model(self):
        """Create a TTS model instance."""
        return SupertoneTTS()
    
    @pytest.fixture
    def temp_output(self):
        """Create a temporary output file."""
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
    
    def test_model_initialization(self, tts_model):
        """Test that the model initializes correctly."""
        assert tts_model is not None
        assert tts_model.text_processor is not None
        assert tts_model.sample_rate == 44100
        assert tts_model.dp_ort is not None
        assert tts_model.text_enc_ort is not None
        assert tts_model.vector_est_ort is not None
        assert tts_model.vocoder_ort is not None
    
    def test_list_voice_styles(self, tts_model):
        """Test listing available voice styles."""
        styles = tts_model.list_voice_styles()
        assert isinstance(styles, list)
        assert len(styles) > 0
        # Check for known styles
        assert "F1" in styles
        assert "M1" in styles
    
    def test_load_voice_style(self, tts_model):
        """Test loading a voice style."""
        style = tts_model.load_voice_style("F1")
        assert style is not None
        assert style.ttl is not None
        assert style.dp is not None
        assert isinstance(style.ttl, np.ndarray)
        assert isinstance(style.dp, np.ndarray)
    
    def test_basic_synthesis(self, tts_model, temp_output):
        """Test basic text-to-speech synthesis."""
        text = "Hello, this is a test."
        result = tts_model.synthesize(
            text=text,
            output_path=temp_output,
            voice_style="F1",
            steps=4  # Use fewer steps for faster testing
        )
        
        assert result == temp_output
        assert os.path.exists(temp_output)
        
        # Verify the audio file
        audio, sr = sf.read(temp_output)
        assert sr == 44100
        assert len(audio) > 0
        assert audio.dtype == np.float32 or audio.dtype == np.float64
    
    def test_synthesis_with_female_voices(self, tts_model, temp_output):
        """Test synthesis with both female voices."""
        for voice in ["F1", "F2"]:
            result = tts_model.synthesize(
                text="Female voice test.",
                output_path=temp_output,
                voice_style=voice,
                steps=4
            )
            assert os.path.exists(result)
            audio, sr = sf.read(result)
            assert len(audio) > 0
    
    def test_synthesis_with_male_voices(self, tts_model, temp_output):
        """Test synthesis with both male voices."""
        for voice in ["M1", "M2"]:
            result = tts_model.synthesize(
                text="Male voice test.",
                output_path=temp_output,
                voice_style=voice,
                steps=4
            )
            assert os.path.exists(result)
            audio, sr = sf.read(result)
            assert len(audio) > 0
    
    def test_synthesis_different_steps(self, tts_model, temp_output):
        """Test synthesis with different inference steps."""
        text = "Testing different quality levels."
        
        for steps in [2, 5, 8, 12]:
            result = tts_model.synthesize(
                text=text,
                output_path=temp_output,
                voice_style="F1",
                steps=steps
            )
            assert os.path.exists(result)
    
    def test_synthesis_different_speeds(self, tts_model, temp_output):
        """Test synthesis with different speed settings."""
        text = "Testing speed variations."
        
        for speed in [0.8, 1.0, 1.2, 1.5]:
            result = tts_model.synthesize(
                text=text,
                output_path=temp_output,
                voice_style="F1",
                steps=4,
                speed=speed
            )
            assert os.path.exists(result)
    
    def test_long_text_synthesis(self, tts_model, temp_output):
        """Test synthesis with longer text."""
        text = (
            "This is a longer text to test the text-to-speech system. "
            "It contains multiple sentences to ensure the model can handle "
            "extended input without issues. The quality should remain consistent "
            "throughout the entire generated audio."
        )
        
        result = tts_model.synthesize(
            text=text,
            output_path=temp_output,
            voice_style="F1",
            steps=5
        )
        
        assert os.path.exists(result)
        audio, sr = sf.read(result)
        # Longer text should produce longer audio
        assert len(audio) > 44100  # At least 1 second
    
    def test_special_characters_handling(self, tts_model, temp_output):
        """Test synthesis with special characters."""
        text = "Testing numbers: 123, and punctuation! Question? Yes."
        
        result = tts_model.synthesize(
            text=text,
            output_path=temp_output,
            voice_style="F1",
            steps=4
        )
        
        assert os.path.exists(result)
    
    def test_invalid_voice_style_fallback(self, tts_model, temp_output):
        """Test that invalid voice styles fall back to default."""
        result = tts_model.synthesize(
            text="Testing fallback.",
            output_path=temp_output,
            voice_style="INVALID_VOICE",
            steps=4
        )
        
        # Should still succeed with fallback to F1
        assert os.path.exists(result)
    
    def test_convenience_function(self, temp_output):
        """Test the convenience function."""
        result = synthesize_with_supertone(
            text="Testing convenience function.",
            output_path=temp_output,
            preset="M1",
            steps=4
        )
        
        assert result == temp_output
        assert os.path.exists(temp_output)


class TestTextProcessing:
    """Test text preprocessing and tokenization."""
    
    @pytest.fixture
    def tts_model(self):
        """Create a TTS model instance."""
        if not MODELS_DIR.exists():
            pytest.skip(SKIP_REASON)
        return SupertoneTTS()
    
    def test_text_processor(self, tts_model):
        """Test text processor."""
        text_ids, text_mask = tts_model.text_processor(["Hello world"])
        
        assert isinstance(text_ids, np.ndarray)
        assert isinstance(text_mask, np.ndarray)
        assert text_ids.shape[0] == 1  # Batch size
        assert text_mask.shape[0] == 1
    
    def test_text_preprocessing(self, tts_model):
        """Test text preprocessing normalization."""
        # Text with special characters that should be normalized
        texts = [
            'Testing - dashes - and quotes "like this"',
            "Email: test@example.com",
            "e.g., for example, i.e., that is",
        ]
        
        for text in texts:
            text_ids, text_mask = tts_model.text_processor([text])
            assert text_ids.shape[1] > 0  # Should have some characters


def test_model_files_exist():
    """Test that all required model files exist."""
    if not MODELS_DIR.exists():
        pytest.skip(SKIP_REASON)
    
    onnx_dir = MODELS_DIR / "onnx"
    voice_styles_dir = MODELS_DIR / "voice_styles"
    
    # Check ONNX models
    assert (onnx_dir / "duration_predictor.onnx").exists()
    assert (onnx_dir / "text_encoder.onnx").exists()
    assert (onnx_dir / "vector_estimator.onnx").exists()
    assert (onnx_dir / "vocoder.onnx").exists()
    assert (onnx_dir / "unicode_indexer.json").exists()
    assert (onnx_dir / "tts.json").exists()
    
    # Check voice styles
    assert (voice_styles_dir / "F1.json").exists()
    assert (voice_styles_dir / "F2.json").exists()
    assert (voice_styles_dir / "M1.json").exists()
    assert (voice_styles_dir / "M2.json").exists()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
