import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.marvis import MarvisTTS

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/marvis")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_basic(output_dir):
    """Test basic synthesis with Marvis."""
    tts = MarvisTTS()
    output_path = output_dir / "marvis_basic.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    tts.synthesize(
        text="Hello, this is a test of the Marvis model.",
        output_path=str(output_path),
        lang_code="en"
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_cloning(output_dir, sample_audio):
    """Test voice cloning with Marvis."""
    tts = MarvisTTS()
    output_path = output_dir / "marvis_cloned.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    tts.synthesize(
        text="This text is spoken using the cloned voice from the sample.",
        output_path=str(output_path),
        ref_audio=sample_audio,
        lang_code="en"
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.parametrize("speed", [0.8, 1.2])
@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_speed(speed, output_dir):
    """Test speed control with Marvis."""
    tts = MarvisTTS()
    output_path = output_dir / f"marvis_speed_{speed}.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    tts.synthesize(
        text=f"Testing Marvis speed at {speed}.",
        output_path=str(output_path),
        speed=speed
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_quantized_check(output_dir):
    """Test Marvis with quantized flag explicitly."""
    tts = MarvisTTS()
    output_path = output_dir / "marvis_quantized.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    # Should use local models/marvis-4bit if exists, or fallback to standard
    tts.synthesize(
        text="Testing Marvis quantized flag.",
        output_path=str(output_path),
        quantized=True
    )
    
    assert output_path.exists()

@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_streaming(output_dir):
    """Test Marvis streaming playback."""
    tts = MarvisTTS()
    output_path = output_dir / "marvis_stream.wav"
    if output_path.exists():
        output_path.unlink()
    
    # Streaming might not produce a file in current implementation
    tts.synthesize(
        text="This is a streaming test.",
        output_path=str(output_path),
        stream=True
    )
    
    # We just verify it ran without error for now

@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_temperature(output_dir):
    """Test Marvis temperature parameter."""
    tts = MarvisTTS()
    output_path = output_dir / "marvis_temp.wav"
    if output_path.exists():
        output_path.unlink()
    
    tts.synthesize(
        text="Testing temperature.",
        output_path=str(output_path),
        temperature=0.9
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.parametrize("lang", ["fr", "de"])
@pytest.mark.skipif(sys.platform != "darwin", reason="Marvis MLX only supported on macOS")
def test_marvis_multilingual(lang, output_dir):
    """Test Marvis multilingual support."""
    tts = MarvisTTS()
    output_path = output_dir / f"marvis_{lang}.wav"
    text = "Bonjour" if lang == "fr" else "Guten Tag"
    
    if output_path.exists():
        output_path.unlink()
        
    tts.synthesize(
        text=text,
        output_path=str(output_path),
        lang_code=lang
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
