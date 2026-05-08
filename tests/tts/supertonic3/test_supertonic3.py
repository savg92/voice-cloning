import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

@pytest.fixture
def mock_onnx():
    with patch("onnxruntime.InferenceSession") as mock_session, \
         patch("onnxruntime.SessionOptions") as mock_options, \
         patch("voice_cloning.tts.supertonic3.snapshot_download") as mock_download, \
         patch("voice_cloning.tts.supertonic3.UnicodeProcessor") as mock_proc:
        
        # Mock Session
        mock_instance = MagicMock()
        mock_session.return_value = mock_instance
        
        # Mock run results
        mock_instance.run.side_effect = [
            [np.array([1.0])], # dur
            [np.zeros((1, 10, 128))], # text_emb
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 128, 10))], # xt (loop)
            [np.zeros((1, 16000))], # wav
        ]
        
        # Mock processor
        mock_proc_instance = MagicMock()
        mock_proc.return_value = mock_proc_instance
        mock_proc_instance.return_value = (np.zeros((1, 10)), np.zeros((1, 1, 10)))
        
        yield {
            "session": mock_session,
            "download": mock_download,
            "instance": mock_instance,
            "processor": mock_proc_instance
        }

@patch("builtins.open", new_callable=MagicMock)
@patch("json.load")
@patch("pathlib.Path.exists")
def test_supertonic3_init(mock_exists, mock_json, mock_open, mock_onnx):
    """Test Supertonic3TTS initialization with mock."""
    from voice_cloning.tts.supertonic3 import Supertonic3TTS
    
    mock_exists.return_value = True
    mock_json.return_value = {
        "ae": {"sample_rate": 24000, "base_chunk_size": 1},
        "ttl": {"chunk_compress_factor": 1, "latent_dim": 128}
    }
    
    tts = Supertonic3TTS(model_dir="/tmp/supertonic3_test")
    assert tts is not None
    assert mock_onnx["session"].call_count == 4

@patch("builtins.open", new_callable=MagicMock)
@patch("json.load")
@patch("pathlib.Path.exists")
@patch("soundfile.write")
def test_supertonic3_synthesis(mock_sf, mock_exists, mock_json, mock_open, mock_onnx):
    """Test Supertonic3TTS synthesis with mock."""
    from voice_cloning.tts.supertonic3 import Supertonic3TTS
    
    mock_exists.return_value = True
    mock_json.side_effect = [
        {"ae": {"sample_rate": 24000, "base_chunk_size": 1}, "ttl": {"chunk_compress_factor": 1, "latent_dim": 128}}, # tts.json
        {"style_ttl": {"data": [0]*128, "dims": [1, 1, 128]}, "style_dp": {"data": [0]*128, "dims": [1, 1, 128]}} # voice style
    ]
    
    tts = Supertonic3TTS(model_dir="/tmp/supertonic3_test")
    output_path = "/tmp/test_output.wav"
    
    result = tts.synthesize(
        text="Hello world",
        output_path=output_path,
        voice="F1",
        lang="en",
        steps=8
    )
    
    assert result == output_path
    assert mock_onnx["instance"].run.call_count == 11 # 1 dur + 1 emb + 8 steps + 1 vocoder
    mock_sf.assert_called_once()

def test_supertonic3_supported_languages():
    """Test supported languages list."""
    from voice_cloning.tts.supertonic3 import Supertonic3TTS
    assert 'en' in Supertonic3TTS.SUPPORTED_LANGUAGES
    assert 'vi' in Supertonic3TTS.SUPPORTED_LANGUAGES
    assert len(Supertonic3TTS.SUPPORTED_LANGUAGES) == 31
