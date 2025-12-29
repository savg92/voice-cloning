import os
import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.asr.canary import CanaryASR, transcribe_to_file

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/canary")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

def test_canary_supported_languages():
    """Test listing and validating languages."""
    asr = CanaryASR()
    langs = asr.get_supported_languages()
    assert "en" in langs
    assert "de" in langs
    assert asr.validate_language("fr")
    assert not asr.validate_language("xx")

def test_canary_transcribe(sample_audio, output_dir):
    """Test basic transcription."""
    asr = CanaryASR()
    
    # Check if dependencies exist
    if not asr.load_model():
        pytest.skip("Canary model could not be loaded (likely missing NeMo/PyAnnote)")
        
    result = asr.transcribe(sample_audio, source_lang="en", target_lang="en")
    assert "text" in result
    assert len(result["text"]) > 0
    assert any(word in result["text"].lower() for word in ["anger", "test", "voice", "hello", "proof", "prove"])

@pytest.mark.parametrize("target_lang", ["fr", "de"])
def test_canary_translation(target_lang, sample_audio):
    """Test speech-to-text translation."""
    asr = CanaryASR()
    if not asr.load_model():
        pytest.skip("Canary model load failed")
        
    result = asr.transcribe(sample_audio, source_lang="en", target_lang=target_lang)
    assert len(result["text"]) > 0

def test_canary_transcribe_to_file(sample_audio, output_dir):
    """Test convenience function transcribe_to_file."""
    output_path = output_dir / "canary_test.txt"
    if output_path.exists():
        output_path.unlink()
        
    # Check dependencies via asr.load_model()
    if not CanaryASR().load_model():
        pytest.skip("Canary model load failed")

    result_path = transcribe_to_file(
        audio_path=sample_audio,
        output_path=str(output_path),
        source_lang="en",
        target_lang="en"
    )
    
    assert Path(result_path).exists()
    assert os.path.getsize(result_path) > 0