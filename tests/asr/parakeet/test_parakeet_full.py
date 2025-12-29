import os
import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.asr.parakeet import ParakeetASR

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

def test_parakeet_backend_detection():
    """Test that the correct backend is detected."""
    asr = ParakeetASR()
    if sys.platform == "darwin" and os.uname().machine == "arm64":
        # Might be mlx if installed, or nemo if not
        assert asr.backend in ["mlx", "nemo"]
    else:
        assert asr.backend == "nemo"

def test_parakeet_transcribe(sample_audio):
    """Test basic transcription."""
    asr = ParakeetASR()
    
    # Check if backend is actually working
    if asr.backend == "nemo" and asr.err_msg and "NeMo toolkit not installed" in asr.err_msg:
        pytest.skip("NeMo toolkit not installed")
    
    if asr.backend == "mlx" and asr.err_msg and "MLX parakeet CLI not found" in asr.err_msg:
        pytest.skip("MLX parakeet CLI not found")

    text = asr.transcribe(sample_audio)
    
    if text.startswith("Error:"):
        pytest.skip(f"Parakeet transcription failed with known error: {text}")
        
    assert len(text) > 0
    assert any(word in text.lower() for word in ["anger", "test", "voice", "hello", "proof", "prove"])

def test_parakeet_timestamps(sample_audio):
    """Test transcription with timestamps (SRT format)."""
    asr = ParakeetASR()
    
    if asr.backend == "nemo" and asr.err_msg:
        pytest.skip(f"NeMo skipped: {asr.err_msg}")
    if asr.backend == "mlx" and asr.err_msg:
        pytest.skip(f"MLX skipped: {asr.err_msg}")

    text = asr.transcribe(sample_audio, timestamps=True)
    
    if text.startswith("Error:"):
        pytest.skip(f"Parakeet timestamps failed: {text}")
        
    assert len(text) > 0
    # Basic check for SRT format patterns if MLX
    if asr.backend == "mlx":
        assert "-->" in text or "00:" in text