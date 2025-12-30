import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.vad.humaware import HumAwareVAD

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

def test_humaware_init():
    """Test VAD model initialization."""
    vad = HumAwareVAD()
    assert vad.model is not None

def test_humaware_detect_basic(sample_audio):
    """Test basic speech detection."""
    vad = HumAwareVAD()
    segments = vad.detect_speech(sample_audio)
    
    assert isinstance(segments, list)
    assert len(segments) > 0
    for seg in segments:
        assert "start" in seg
        assert "end" in seg
        assert seg["end"] > seg["start"]

@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_humaware_thresholds(threshold, sample_audio):
    """Test different sensitivity thresholds."""
    vad = HumAwareVAD()
    segments = vad.detect_speech(sample_audio, threshold=threshold)
    
    # We just want to ensure it doesn't crash and returns something sensible
    assert isinstance(segments, list)
    # Higher threshold usually means fewer segments or shorter segments
    # but we don't strictly assert that here as it depends on the audio

def test_humaware_parameters(sample_audio):
    """Test VAD with custom parameters."""
    vad = HumAwareVAD()
    segments = vad.detect_speech(
        sample_audio,
        min_speech_duration_ms=500,
        min_silence_duration_ms=200,
        speech_pad_ms=50
    )
    
    assert isinstance(segments, list)
    if len(segments) > 0:
        # Check that durations are at least roughly matching our constraint
        # (with padding, it might be more)
        for seg in segments:
            duration = seg["end"] - seg["start"]
            assert duration > 0.4 # roughly 0.5s minus small resample/math diffs
