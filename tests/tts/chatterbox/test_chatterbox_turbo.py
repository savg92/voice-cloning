import os
import pytest
import torch
from pathlib import Path
from src.voice_cloning.tts.chatterbox_turbo import synthesize_with_chatterbox_turbo

@pytest.fixture
def output_dir():
    path = Path("tests/output/chatterbox_turbo")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

def test_chatterbox_turbo_pytorch_basic(output_dir):
    """Test basic synthesis with Turbo PyTorch backend."""
    output_path = output_dir / "test_turbo_pt_basic.wav"
    text = "Hello, this is a test of Chatterbox Turbo PyTorch."
    
    synthesize_with_chatterbox_turbo(
        text=text,
        output_wav=str(output_path),
        use_mlx=False,
        device="cpu"
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

@pytest.mark.skipif(not torch.backends.mps.is_available(), 
                    reason="MLX requires Apple Silicon (MPS)")
def test_chatterbox_turbo_mlx_basic(output_dir):
    """Test basic synthesis with Turbo MLX backend."""
    output_path = output_dir / "test_turbo_mlx_basic.wav"
    text = "Hello, this is a test of Chatterbox Turbo MLX."
    
    synthesize_with_chatterbox_turbo(
        text=text,
        output_wav=str(output_path),
        use_mlx=True
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_chatterbox_turbo_cloning(output_dir, sample_audio):
    """Test voice cloning with Turbo model."""
    output_path = output_dir / "test_turbo_cloning.wav"
    text = "I am cloning your voice with Turbo speed."
    
    synthesize_with_chatterbox_turbo(
        text=text,
        output_wav=str(output_path),
        source_wav=sample_audio,
        use_mlx=False,
        device="cpu"
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_chatterbox_turbo_multilingual_skip(output_dir):
    """Verify that multilingual calls are handled gracefully or documented as English-only."""
    output_path = output_dir / "test_turbo_multi_fallback.wav"
    text = "Testing fallback logic."
    
    # This should work but warn and use English
    synthesize_with_chatterbox_turbo(
        text=text,
        output_wav=str(output_path),
        language="fr",
        multilingual=True,
        use_mlx=False,
        device="cpu"
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0