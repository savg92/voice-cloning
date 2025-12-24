import pytest
import gradio as gr
from unittest.mock import patch, MagicMock
from src.voice_cloning.ui.vad_tab import create_vad_tab, detect_speech_segments

def test_create_vad_tab_returns_component():
    """Test that create_vad_tab returns a Gradio Component."""
    with gr.Blocks():
        tab = create_vad_tab()
        assert isinstance(tab, gr.blocks.BlockContext) or isinstance(tab, gr.components.Component)

@patch("src.voice_cloning.vad.humaware.HumAwareVAD")
def test_detect_speech_segments(MockVAD):
    """Test that VAD analysis calls the correct backend with params."""
    mock_instance = MockVAD.return_value
    mock_instance.detect_speech.return_value = [{'start': 0.0, 'end': 1.0}]
    
    output = detect_speech_segments("audio.wav", 0.5, 250, 100, 30)
    
    MockVAD.assert_called_once()
    mock_instance.detect_speech.assert_called_with(
        "audio.wav",
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        speech_pad_ms=30
    )
    assert "0.0" in output
    assert "1.0" in output

def test_detect_speech_segments_missing_audio():
    """Test that VAD raises error if audio is missing."""
    with pytest.raises(gr.Error, match="Please upload an audio file"):
        detect_speech_segments(None, 0.5, 250, 100, 30)
