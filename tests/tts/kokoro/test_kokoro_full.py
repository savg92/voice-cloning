import os
import sys
import logging
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.voice_cloning.tts.kokoro import synthesize_speech

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

@pytest.mark.parametrize("lang_code", TEST_CASES.keys())
@pytest.mark.parametrize("use_mlx", [True, False] if sys.platform == "darwin" else [False])
def test_kokoro_all_languages(lang_code, use_mlx):
    """Test Kokoro with all supported languages and backends."""
    text = TEST_CASES[lang_code]
    backend = "MLX" if use_mlx else "PyTorch"
    output_dir = Path("outputs/tests/kokoro")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"kokoro_{lang_code}_{backend.lower()}.wav"
    output_path = output_dir / filename
    
    if output_path.exists():
        output_path.unlink()
        
    logger.info(f"Testing {lang_code} ({backend})...")
    
    try:
        result = synthesize_speech(
            text=text,
            output_path=str(output_path),
            lang_code=lang_code,
            use_mlx=use_mlx
        )
        
        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        logger.info(f"✅ Success: {filename}")
        
    except Exception as e:
        pytest.fail(f"❌ Failed {lang_code} ({backend}): {e}")

def test_kokoro_streaming():
    """Test Kokoro streaming option."""
    text = "This is a short test for streaming."
    output_path = "outputs/tests/kokoro/kokoro_stream_test.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    logger.info("Testing streaming (PyTorch)...")
    synthesize_speech(
        text=text,
        output_path=output_path,
        lang_code="a",
        stream=True,
        use_mlx=False
    )
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

def test_kokoro_speed():
    """Test Kokoro speed control."""
    text = "Testing speed control at 1.5x."
    output_path = "outputs/tests/kokoro/kokoro_speed_test.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    synthesize_speech(
        text=text,
        output_path=output_path,
        lang_code="a",
        speed=1.5
    )
    assert os.path.exists(output_path)

if __name__ == "__main__":
    # If run directly, run a quick smoke test for US English
    test_kokoro_all_languages("a", use_mlx=(sys.platform == "darwin"))
