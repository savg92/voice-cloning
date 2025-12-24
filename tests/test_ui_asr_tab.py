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
    """Test that Whisper transcription calls the correct backend with params."""
    mock_instance = MockWhisper.return_value
    mock_instance.transcribe.return_value = "Whisper transcript"
    
    output = transcribe_speech(
        "Whisper", "audio.wav", 
        "openai/whisper-tiny", "en", "transcribe", False, True,
        False, "en", "en"
    )
    
    MockWhisper.assert_called_with(model_id="openai/whisper-tiny", use_mlx=False)
    mock_instance.transcribe.assert_called_with("audio.wav", lang="en", task="transcribe", timestamps=True)
    assert output == "Whisper transcript"

@patch("src.voice_cloning.asr.granite.transcribe_file")
@patch("builtins.open", new_callable=MagicMock)
def test_transcribe_speech_granite(mock_open, mock_transcribe):
    """Test that Granite transcription calls the correct backend."""
    # Mock open().read()
    mock_open.return_value.__enter__.return_value.read.return_value = "Granite transcript"
    
    output = transcribe_speech(
        "Granite", "audio.wav",
        "", "", "", False, False, False, "", ""
    )
    
    mock_transcribe.assert_called_once()
    assert output == "Granite transcript"

def test_transcribe_speech_missing_audio():
    """Test that transcription raises error if audio is missing."""
    with pytest.raises(gr.Error, match="Please upload an audio file"):
        transcribe_speech("Whisper", None, "", "", "", False, False, False, "", "")