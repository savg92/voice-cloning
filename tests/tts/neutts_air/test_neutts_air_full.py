import os
import sys
import pytest
from pathlib import Path
import soundfile as sf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.neutts_air import NeuTTSAirTTS, synthesize_with_neutts_air

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/neutts_air")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def model_exists():
    path = Path("models/neuttsair")
    if not path.exists():
        pytest.skip("NeuTTS Air models not found in models/neuttsair")
    return True

@pytest.fixture
def dave_sample():
    audio = Path("samples/neutts_air/dave.wav")
    text = Path("samples/neutts_air/dave.txt")
    if not audio.exists() or not text.exists():
        # Create dave.txt if missing but audio exists
        if audio.exists() and not text.exists():
             with open(text, "w") as f:
                 f.write("This is Dave's reference voice sample.")
        else:
             pytest.skip("NeuTTS Air sample 'dave' not found")
    return str(audio), str(text)

def test_neutts_air_cloning(model_exists, dave_sample, output_dir):
    """Test voice cloning with NeuTTS Air."""
    audio_ref, text_ref = dave_sample
    output_path = output_dir / "neutts_cloned.wav"
    text = "Hello, I am testing voice cloning with Neu TTS Air."
    
    synthesize_with_neutts_air(
        text=text,
        output_path=str(output_path),
        ref_audio=audio_ref,
        ref_text=text_ref
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    
    info = sf.info(str(output_path))
    assert info.samplerate == 24000 # NeuTTS is 24k

def test_neutts_air_auto_ref_text(model_exists, dave_sample, output_dir):
    """Test NeuTTS Air auto-detection of reference text file."""
    audio_ref, _ = dave_sample
    output_path = output_dir / "neutts_auto_ref.wav"
    
    tts = NeuTTSAirTTS()
    tts.synthesize(
        text="Testing auto-detection of reference text.",
        output_path=str(output_path),
        ref_audio_path=audio_ref
        # ref_text_path is None, should find .txt automatically
    )
    
    assert output_path.exists()
