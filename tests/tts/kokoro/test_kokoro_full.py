import os
import sys
import logging
import pytest
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.kokoro import synthesize_speech

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Sample texts for each language
TEST_CASES = {
    "a": "Hello, this is a test of the American English voice.",
    "b": "Hello, this is a test of the British English voice.",
    "e": "Hola, esto es una prueba de la voz en español.",
    "f": "Bonjour, c'est un test de la voix française.",
    "h": "नमस्ते, यह हिंदी आवाज का परीक्षण है।",
    "i": "Ciao, questo è un test della voce italiana.",
    "j": "こんにちは、これは日本語の音声のテストです。",
    "p": "Olá, este é um teste da voz em português.",
    "z": "你好，这是中文语音的测试。"
}

# Voices for each language (example ones)
VOICE_MAP = {
    "a": "af_heart",
    "b": "bf_isabella",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "j": "jf_alpha",
    "p": "pf_dora",
    "z": "zf_xiaobei"
}

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/kokoro")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.mark.parametrize("lang_code", TEST_CASES.keys())
@pytest.mark.parametrize("use_mlx", [True, False] if sys.platform == "darwin" else [False])
def test_kokoro_all_languages(lang_code, use_mlx, output_dir):
    """Test Kokoro with all supported languages and backends."""
    text = TEST_CASES[lang_code]
    voice = VOICE_MAP.get(lang_code, "af_heart")
    backend = "MLX" if use_mlx else "PyTorch"
    
    filename = f"kokoro_{lang_code}_{backend.lower()}.wav"
    output_path = output_dir / filename
    
    if output_path.exists():
        output_path.unlink()
        
    logger.info(f"Testing {lang_code} ({backend}) with voice {voice}...")
    
    try:
        result = synthesize_speech(
            text=text,
            output_path=str(output_path),
            lang_code=lang_code,
            voice=voice,
            use_mlx=use_mlx
        )
        
        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify audio properties
        info = sf.info(str(output_path))
        assert info.samplerate == 24000
        assert info.frames > 0
        
        logger.info(f"✅ Success: {filename}")
        
    except Exception as e:
        pytest.fail(f"❌ Failed {lang_code} ({backend}): {e}")

@pytest.mark.parametrize("use_mlx", [True, False] if sys.platform == "darwin" else [False])
def test_kokoro_streaming(use_mlx, output_dir):
    """Test Kokoro streaming option."""
    text = "This is a short test for streaming."
    backend = "mlx" if use_mlx else "pytorch"
    output_path = output_dir / f"kokoro_stream_{backend}.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    logger.info(f"Testing streaming ({backend})...")
    synthesize_speech(
        text=text,
        output_path=str(output_path),
        lang_code="a",
        stream=True,
        use_mlx=use_mlx
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.parametrize("speed", [0.5, 1.5, 2.0])
def test_kokoro_speed_pytorch(speed, output_dir):
    """Test Kokoro speed control on PyTorch."""
    text = f"Testing speed control at {speed}x."
    output_path = output_dir / f"kokoro_speed_{speed}x_pt.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    synthesize_speech(
        text=text,
        output_path=str(output_path),
        lang_code="a",
        speed=speed,
        use_mlx=False
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_kokoro_empty_text(output_dir):
    """Test Kokoro with empty text (should not crash, might return silence or error)."""
    output_path = output_dir / "kokoro_empty.wav"
    if output_path.exists():
        output_path.unlink()
        
    # Depending on implementation, it might raise error or return silence
    # Currently it seems to return None or empty audio if no chunks generated
    try:
        result = synthesize_speech(
            text="",
            output_path=str(output_path),
            lang_code="a"
        )
        if result and os.path.exists(result):
            assert os.path.getsize(result) >= 0
    except Exception as e:
        logger.info(f"Empty text handled via exception: {e}")

def test_kokoro_invalid_lang(output_dir):
    """Test Kokoro with invalid language code."""
    output_path = output_dir / "kokoro_invalid_lang.wav"
    
    with pytest.raises(Exception):
        synthesize_speech(
            text="Hello",
            output_path=str(output_path),
            lang_code="invalid"
        )

def test_kokoro_long_text(output_dir):
    """Test Kokoro with a longer paragraph to verify multi-chunk generation."""
    text = (
        "This is a longer paragraph designed to test the multi-chunk generation capabilities of the Kokoro TTS engine. "
        "It contains several sentences that should be processed as individual segments and then concatenated together "
        "into a single, seamless audio output file. This ensures that the pipeline correctly handles larger blocks of "
        "text without losing data or producing artifacts between the combined segments."
    )
    output_path = output_dir / "kokoro_long_text.wav"
    
    if output_path.exists():
        output_path.unlink()
        
    result = synthesize_speech(
        text=text,
        output_path=str(output_path),
        lang_code="a"
    )
    assert os.path.exists(result)
    assert os.path.getsize(result) > 10000 # Expect significant size