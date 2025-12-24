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

@patch("src.voice_cloning.tts.chatterbox.synthesize_with_chatterbox")
@patch("tempfile.mktemp")
def test_generate_speech_chatterbox(mock_mktemp, mock_chatterbox):
    """Test that generating speech with Chatterbox calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    
    output = generate_speech("Chatterbox", "Hello", reference_audio="ref.wav")
    
    mock_chatterbox.assert_called_once()
    assert output == "output_path.wav"
    # Verify reference was passed
    args, kwargs = mock_chatterbox.call_args
    assert kwargs['source_wav'] == "ref.wav"

@patch("src.voice_cloning.tts.marvis.MarvisTTS")
@patch("tempfile.mktemp")
def test_generate_speech_marvis(mock_mktemp, MockMarvisTTS):
    """Test that generating speech with Marvis calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    mock_instance = MockMarvisTTS.return_value
    
    output = generate_speech("Marvis", "Hello", reference_audio="ref.wav")
    
    MockMarvisTTS.assert_called_once()
    mock_instance.synthesize.assert_called_once()
    assert output == "output_path.wav"
    # Verify reference was passed
    args, kwargs = mock_instance.synthesize.call_args
    assert kwargs['ref_audio'] == "ref.wav"

def test_generate_speech_missing_reference():
    """Test that voice cloning models raise error if reference is missing."""
    with pytest.raises(gr.Error, match="Reference audio is required"):
        generate_speech("Chatterbox", "Hello", reference_audio=None)

@patch("src.voice_cloning.tts.cosyvoice.synthesize_speech")
@patch("tempfile.mktemp")
def test_generate_speech_cosyvoice(mock_mktemp, mock_cosyvoice):
    """Test that generating speech with CosyVoice calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    
    output = generate_speech("CosyVoice", "Hello", reference_audio="ref.wav")
    
    mock_cosyvoice.assert_called_once()
    assert output == "output_path.wav"
    # Verify reference was passed
    args, kwargs = mock_cosyvoice.call_args
    assert kwargs['ref_audio_path'] == "ref.wav"

def test_generate_speech_unknown_model():
    """Test that generating speech with an unknown model raises Error."""
    with pytest.raises(gr.Error):
        generate_speech("Unknown", "Hello")