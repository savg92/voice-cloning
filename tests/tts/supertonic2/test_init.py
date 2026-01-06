import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

def test_supertonic2_import():
    """Test that Supertonic2TTS can be imported."""
    try:
        from voice_cloning.tts.supertonic2 import Supertonic2TTS
        assert Supertonic2TTS is not None
    except ImportError:
        pytest.fail("Could not import Supertonic2TTS")

def test_supertonic2_init():
    """Test Supertonic2TTS initialization."""
    from voice_cloning.tts.supertonic2 import Supertonic2TTS
    # This should fail if models are missing or initialization fails
    # For Red Phase, we just want to see it fail because the module doesn't exist
    tts = Supertonic2TTS()
    assert tts is not None
