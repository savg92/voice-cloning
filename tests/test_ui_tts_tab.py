import pytest
import gradio as gr
from unittest.mock import patch, MagicMock
from src.voice_cloning.ui.tts_tab import create_tts_tab, generate_speech

def test_create_tts_tab_returns_component():
    """Test that create_tts_tab returns a Gradio Component (likely a Column or Group)."""
    with gr.Blocks():
        tab = create_tts_tab()
        # It usually returns a Container/Column/Group which inherits from BlockContext/Component
        assert isinstance(tab, gr.blocks.BlockContext) or isinstance(tab, gr.components.Component)

@patch("src.voice_cloning.tts.kokoro.synthesize_speech")
@patch("tempfile.mktemp")
def test_generate_speech_kokoro(mock_mktemp, mock_kokoro):
    """Test that generating speech with Kokoro calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    mock_kokoro.return_value = None # The function returns None (saves to file)
    
    output = generate_speech("Kokoro", "Hello World")
    
    mock_kokoro.assert_called_once()
    assert output == "output_path.wav"

@patch("src.voice_cloning.tts.kitten_nano.KittenNanoTTS")
@patch("tempfile.mktemp")
def test_generate_speech_kitten(mock_mktemp, MockKittenNanoTTS):
    """Test that generating speech with Kitten calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    
    # Setup mock instance
    mock_instance = MockKittenNanoTTS.return_value
    mock_instance.synthesize_to_file.return_value = None
    
    output = generate_speech("Kitten", "Hello World")
    
    MockKittenNanoTTS.assert_called_once()
    mock_instance.synthesize_to_file.assert_called_once()
    assert output == "output_path.wav"

def test_generate_speech_unknown_model():
    """Test that generating speech with an unknown model raises Error."""
    with pytest.raises(gr.Error):
        generate_speech("Unknown", "Hello")
