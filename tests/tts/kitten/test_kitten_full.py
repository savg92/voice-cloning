import sys
import pytest
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.kitten_nano import KittenNanoTTS

# Kitten Nano Models
MODELS = [
    "KittenML/kitten-tts-nano-0.1",
    "KittenML/kitten-tts-nano-0.2"
]

# Common voices
VOICES = [
    "expr-voice-2-f",
    "expr-voice-2-m",
    "expr-voice-3-f",
    "expr-voice-3-m",
    "expr-voice-4-f",
    "expr-voice-4-m",
    "expr-voice-5-f",
    "expr-voice-5-m",
]

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/kitten")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.mark.parametrize("model_id", MODELS)
def test_kitten_versions(model_id, output_dir):
    """Test multiple versions of Kitten Nano."""
    tts = KittenNanoTTS(model_id=model_id)
    text = f"This is a test of {model_id}."
    filename = model_id.split("/")[-1] + ".wav"
    output_path = output_dir / filename
    
    if output_path.exists():
        output_path.unlink()
        
    result = tts.synthesize_to_file(text, str(output_path))
    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    
    info = sf.info(str(output_path))
    assert info.samplerate == 24000

@pytest.mark.parametrize("voice", VOICES[:3]) # Test first 3 voices to save time
def test_kitten_voices(voice, output_dir):
    """Test multiple voices with the latest Kitten Nano (0.2)."""
    tts = KittenNanoTTS(model_id="KittenML/kitten-tts-nano-0.2")
    text = f"This is a test of voice {voice}."
    output_path = output_dir / f"kitten_voice_{voice}.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    tts.synthesize_to_file(text, str(output_path), voice=voice)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_kitten_streaming(output_dir):
    """Test streaming playback with Kitten Nano."""
    tts = KittenNanoTTS()
    text = "This is a streaming test for Kitten TTS."
    output_path = output_dir / "kitten_stream.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    # We use stream=True. This normally plays audio AND saves to file in our wrapper.
    tts.synthesize_to_file(text, str(output_path), stream=True)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.parametrize("speed", [0.8, 1.2])
def test_kitten_speed(speed, output_dir):
    """Test speed control with Kitten Nano."""
    tts = KittenNanoTTS()
    text = f"This is a test at speed {speed}."
    output_path = output_dir / f"kitten_speed_{speed}.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    tts.synthesize_to_file(text, str(output_path), speed=speed)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_kitten_long_text(output_dir):
    """Test longer text with Kitten Nano (checks sentence splitting if used)."""
    text = (
        "This is a longer text for Kitten TTS. It contains multiple sentences. "
        "We want to ensure that the wrapper handles larger blocks of text correctly "
        "and produces a single coherent audio file at the end."
    )
    output_path = output_dir / "kitten_long.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    tts = KittenNanoTTS()
    tts.synthesize_to_file(text, str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 20000 # Expect decent size
