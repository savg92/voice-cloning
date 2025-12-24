import pytest
import gradio as gr
from unittest.mock import patch, MagicMock
from src.voice_cloning.ui.asr_tab import create_asr_tab, transcribe_speech

def test_create_asr_tab_returns_component():
    """Test that create_asr_tab returns a Gradio Component."""
    with gr.Blocks():
        tab = create_asr_tab()
        assert isinstance(tab, gr.blocks.BlockContext) or isinstance(tab, gr.components.Component)

@patch("src.voice_cloning.asr.whisper.WhisperASR")
def test_transcribe_speech_whisper(MockWhisper):
    """Test that Whisper transcription calls the correct backend."""
    mock_instance = MockWhisper.return_value
    mock_instance.transcribe.return_value = "Whisper transcript"
    
    output = transcribe_speech("Whisper", "audio.wav")
    
    MockWhisper.assert_called_once()
    mock_instance.transcribe.assert_called_with("audio.wav")
    assert output == "Whisper transcript"

@patch("src.voice_cloning.asr.parakeet.ParakeetASR")
def test_transcribe_speech_parakeet(MockParakeet):
    """Test that Parakeet transcription calls the correct backend."""
    mock_instance = MockParakeet.return_value
    mock_instance.transcribe.return_value = "Parakeet transcript"
    
    output = transcribe_speech("Parakeet", "audio.wav")
    
    MockParakeet.assert_called_once()
    mock_instance.transcribe.assert_called_with("audio.wav")
    assert output == "Parakeet transcript"

@patch("src.voice_cloning.asr.canary.CanaryASR")
def test_transcribe_speech_canary(MockCanary):
    """Test that Canary transcription calls the correct backend."""
    mock_instance = MockCanary.return_value
    mock_instance.transcribe.return_value = {'text': "Canary transcript"}
    
    output = transcribe_speech("Canary", "audio.wav")
    
    MockCanary.assert_called_once()
    mock_instance.load_model.assert_called_once()
    mock_instance.transcribe.assert_called_with("audio.wav")
    assert output == "Canary transcript"

def test_transcribe_speech_missing_audio():
    """Test that transcription raises error if audio is missing."""
    with pytest.raises(gr.Error, match="Please upload an audio file"):
        transcribe_speech("Whisper", None)
