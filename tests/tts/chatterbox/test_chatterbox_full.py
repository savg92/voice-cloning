import sys
import pytest
import torch
from pathlib import Path

try:
    import chatterbox
    HAS_CHATTERBOX = True
except ImportError:
    HAS_CHATTERBOX = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.chatterbox import ChatterboxWrapper, synthesize_with_chatterbox

pytestmark = pytest.mark.skipif(not HAS_CHATTERBOX, reason="chatterbox-tts package not installed")


@pytest.fixture
def output_dir():
    path = Path("outputs/tests/chatterbox")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(), 
                    reason="Requires GPU/MPS for reasonable performance")
def test_chatterbox_basic_pytorch(output_dir):
    """Test basic synthesis with PyTorch backend."""
    output_path = output_dir / "chatterbox_basic_pt.wav"
    text = "This is a basic test of Chatterbox with PyTorch."
    
    synthesize_with_chatterbox(
        text=text,
        output_wav=str(output_path),
        use_mlx=False
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.skipif(sys.platform != "darwin", reason="MLX only supported on macOS")
def test_chatterbox_mlx_check(output_dir, sample_audio):
    """Test Chatterbox with MLX backend (checks for support)."""
    output_path = output_dir / "chatterbox_mlx.wav"
    text = "Testing Chatterbox with MLX backend."
    
    try:
        synthesize_with_chatterbox(
            text=text,
            output_wav=str(output_path),
            source_wav=sample_audio,
            use_mlx=True
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    except RuntimeError as e:
        if "not yet supported" in str(e).lower() or "not supported" in str(e).lower():
            pytest.skip(f"MLX Chatterbox not supported in current environment: {e}")
        else:
            raise

def test_chatterbox_cloning(output_dir, sample_audio):
    """Test zero-shot voice cloning."""
    output_path = output_dir / "chatterbox_cloned.wav"
    text = "This is a test of zero-shot voice cloning using Chatterbox."
    
    synthesize_with_chatterbox(
        text=text,
        output_wav=str(output_path),
        source_wav=sample_audio,
        use_mlx=False # Use PT for reliability in test
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.parametrize("lang", ["fr", "es", "zh"])
def test_chatterbox_multilingual(lang, output_dir):
    """Test multilingual support."""
    texts = {
        "fr": "Bonjour, c'est un test de Chatterbox en français.",
        "es": "Hola, esta es una prueba de Chatterbox en español.",
        "zh": "你好，这是 Chatterbox 的中文测试。"
    }
    
    output_path = output_dir / f"chatterbox_{lang}.wav"
    
    synthesize_with_chatterbox(
        text=texts[lang],
        output_wav=str(output_path),
        language=lang,
        multilingual=True,
        use_mlx=False
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_chatterbox_controls(output_dir):
    """Test exaggeration and CFG weight controls."""
    output_path = output_dir / "chatterbox_controls.wav"
    text = "Testing controls."
    
    synthesize_with_chatterbox(
        text=text,
        output_wav=str(output_path),
        exaggeration=0.8,
        cfg_weight=0.3,
        use_mlx=False
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
