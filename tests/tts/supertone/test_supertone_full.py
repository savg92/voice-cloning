import sys
import pytest
from pathlib import Path
import soundfile as sf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.supertone import SupertoneTTS, synthesize_with_supertone

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/supertone")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def model_exists():
    path = Path("models/supertonic/onnx")
    if not path.exists():
        pytest.skip("Supertone ONNX models not found in models/supertonic/onnx")
    return True

def test_supertone_list_styles(model_exists):
    """Test listing available voice styles."""
    tts = SupertoneTTS()
    styles = tts.list_voice_styles()
    assert len(styles) > 0
    assert "F1" in styles

@pytest.mark.parametrize("style", ["F1", "M1"])
def test_supertone_styles(style, model_exists, output_dir):
    """Test synthesis with different voice styles."""
    output_path = output_dir / f"supertone_{style}.wav"
    text = f"This is a test of Supertone style {style}."
    
    synthesize_with_supertone(
        text=text,
        output_path=str(output_path),
        preset=style
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    
    info = sf.info(str(output_path))
    assert info.samplerate == 44100 # Supertone models in use are 44.1k

def test_supertone_streaming(model_exists, output_dir):
    """Test pseudo-streaming playback with Supertone."""
    output_path = output_dir / "supertone_stream.wav"
    text = "This is a streaming test for Supertone. It processes sentences individually."
    
    tts = SupertoneTTS()
    tts.synthesize(
        text=text,
        output_path=str(output_path),
        stream=True
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.parametrize("speed", [0.8, 1.5])
def test_supertone_speed(speed, model_exists, output_dir):
    """Test speed control with Supertone."""
    output_path = output_dir / f"supertone_speed_{speed}.wav"
    text = f"Testing speed at {speed}x."
    
    tts = SupertoneTTS()
    tts.synthesize(
        text=text,
        output_path=str(output_path),
        speed=speed
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_supertone_parameters(model_exists, output_dir):
    """Test Supertone advanced parameters (steps)."""
    output_path = output_dir / "supertone_params.wav"
    text = "Testing inference steps."
    
    tts = SupertoneTTS()
    # Use fewer steps to be faster, but non-default to verify it accepts it
    tts.synthesize(
        text=text,
        output_path=str(output_path),
        steps=4
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
