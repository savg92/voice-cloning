import os
import sys
import pytest
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.asr.whisper import WhisperASR, transcribe_to_file

@pytest.fixture
def output_dir():
    path = Path("outputs/tests/whisper")
    path.mkdir(parents=True, exist_ok=True)
    return path

@pytest.fixture
def sample_audio():
    path = Path("samples/anger.wav")
    if not path.exists():
        pytest.skip("Reference audio samples/anger.wav not found")
    return str(path)

def test_whisper_basic_pytorch(output_dir, sample_audio):
    """Test basic transcription with PyTorch backend (tiny model for speed)."""
    output_path = output_dir / "whisper_basic_pt.txt"
    
    # Use tiny model for fast testing
    model = WhisperASR(model_id="openai/whisper-tiny", use_mlx=False)
    text = model.transcribe(sample_audio)
    
    assert len(text) > 0
    # The sample text is about proof/prove
    assert any(word in text.lower() for word in ["anger", "test", "voice", "hello", "proof", "prove"])

@pytest.mark.skipif(sys.platform != "darwin", reason="MLX only supported on macOS")
def test_whisper_mlx(output_dir, sample_audio):
    """Test transcription with MLX backend."""
    # MLX whisper might be slow or require specific setup
    try:
        import mlx_whisper
    except ImportError:
        pytest.skip("mlx-whisper package not installed")
        
    model = WhisperASR(use_mlx=True)
    text = model.transcribe(sample_audio)
    
    assert len(text) > 0

@pytest.mark.parametrize("lang", ["fr", "es"])
def test_whisper_multilingual(lang, sample_audio):
    """Test multilingual transcription."""
    model = WhisperASR(model_id="openai/whisper-tiny", use_mlx=False)
    # The sample is English, but we can tell Whisper it's another language to test the pipeline
    text = model.transcribe(sample_audio, lang=lang)
    assert len(text) > 0

def test_whisper_translation(sample_audio):
    """Test speech-to-text translation."""
    model = WhisperASR(model_id="openai/whisper-tiny", use_mlx=False)
    # Translate English sample to French text
    text = model.transcribe(sample_audio, task="translate", lang="fr")
    assert len(text) > 0

def test_whisper_to_file(output_dir, sample_audio):
    """Test convenience function transcribe_to_file."""
    output_path = output_dir / "whisper_file_test.txt"
    if output_path.exists():
        output_path.unlink()
        
    transcribe_to_file(
        audio_path=sample_audio,
        output_path=str(output_path),
        model_id="openai/whisper-tiny"
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_whisper_timestamps(sample_audio):
    """Test transcription with timestamps."""
    model = WhisperASR(model_id="openai/whisper-tiny", use_mlx=False)
    # Testing that it doesn't crash and returns text
    text = model.transcribe(sample_audio, timestamps=True)
    assert len(text) > 0

def test_whisper_to_file_timestamps(output_dir, sample_audio):
    """Test transcribe_to_file with timestamps."""
    output_path = output_dir / "whisper_timestamps.txt"
    if output_path.exists():
        output_path.unlink()
        
    transcribe_to_file(
        audio_path=sample_audio,
        output_path=str(output_path),
        model_id="openai/whisper-tiny",
        timestamps=True,
        use_mlx=False
    )
    
    assert output_path.exists()
    content = output_path.read_text()
    assert "--- Timestamps ---" in content

    