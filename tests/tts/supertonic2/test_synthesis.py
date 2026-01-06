import os
import sys
import pytest
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.supertonic2 import Supertonic2TTS

@pytest.fixture
def tts():
    return Supertonic2TTS()

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/supertonic2")
    path.mkdir(parents=True, exist_ok=True)
    return path

TEST_CASES = [
    ("en", "Hello, this is a test of Supertonic 2 in English."),
    ("ko", "안녕하세요, 슈퍼토닉 2 한국어 테스트입니다."),
    ("es", "Hola, esto es una prueba de Supertonic 2 en español."),
    ("fr", "Bonjour, ceci est un test de Supertonic 2 en français."),
    ("pt", "Olá, este é um teste do Supertonic 2 em português.")
]

@pytest.mark.parametrize("lang, text", TEST_CASES)
def test_synthesis_languages(tts, output_dir, lang, text):
    """Test synthesis in different languages."""
    output_path = output_dir / f"test_{lang}.wav"
    if output_path.exists():
        output_path.unlink()
    
    result = tts.synthesize(text, str(output_path), lang_code=lang)
    
    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    
    # Verify audio
    info = sf.info(str(output_path))
    assert info.samplerate == 44100
    assert info.frames > 0

def test_synthesis_speed(tts, output_dir):
    """Test synthesis with different speeds."""
    text = "Testing speed control."
    
    # Normal speed
    out_normal = output_dir / "speed_1.0.wav"
    tts.synthesize(text, str(out_normal), speed=1.0)
    dur_normal = sf.info(str(out_normal)).duration
    
    # Fast speed
    out_fast = output_dir / "speed_2.0.wav"
    tts.synthesize(text, str(out_fast), speed=2.0)
    dur_fast = sf.info(str(out_fast)).duration
    
    # Fast should be significantly shorter than normal
    assert dur_fast < dur_normal

def test_synthesis_steps(tts, output_dir):
    """Test synthesis with different step counts."""
    text = "Testing inference steps."
    
    # Low steps (fast)
    out_low = output_dir / "steps_4.wav"
    tts.synthesize(text, str(out_low), steps=4)
    assert out_low.exists()
    
    # High steps (higher quality)
    out_high = output_dir / "steps_16.wav"
    tts.synthesize(text, str(out_high), steps=16)
    assert out_high.exists()

def test_all_voices(tts, output_dir):
    """Test synthesis with all available voices."""
    text = "Checking voice."
    voices = tts.list_voice_styles()
    
    # Test first 2 and last 2 to save time if many, but there are only 10
    for v in voices:
        out = output_dir / f"voice_{v}.wav"
        tts.synthesize(text, str(out), voice_style=v)
        assert out.exists()

def test_synthesis_streaming(tts, output_dir):
    """Test pseudo-streaming synthesis."""
    text = "This is a test for streaming. It has multiple sentences. To verify it works."
    output_path = output_dir / "test_streaming.wav"
    if output_path.exists():
        output_path.unlink()
        
    result = tts.synthesize(text, str(output_path), stream=True)
    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
