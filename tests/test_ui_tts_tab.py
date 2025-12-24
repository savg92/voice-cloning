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
    """Test that generating speech with Kokoro calls the correct function with params."""
    mock_mktemp.return_value = "output_path.wav"
    
    output = generate_speech(
        "Kokoro", "Hello World", None, # model, text, ref
        1.0, True, # speed, mlx
        "af_heart", "a", # kokoro
        "v", # kitten
        0.7, 0.5, "en", False, # chatter
        0.7, 0.95, True, # marvis
        "" # cosy
    )
    
    mock_kokoro.assert_called_with(
        text="Hello World", 
        output_path="output_path.wav",
        voice="af_heart",
        lang_code="a",
        speed=1.0,
        use_mlx=True
    )
    assert output == "output_path.wav"

@patch("src.voice_cloning.tts.kitten_nano.KittenNanoTTS")
@patch("tempfile.mktemp")
def test_generate_speech_kitten(mock_mktemp, MockKittenNanoTTS):
    """Test that generating speech with Kitten calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    mock_instance = MockKittenNanoTTS.return_value
    
    output = generate_speech(
        "Kitten", "Hello World", None,
        1.2, False,
        "v", "l",
        "expr-voice-4-f",
        0.7, 0.5, "en", False,
        0.7, 0.95, True,
        ""
    )
    
    MockKittenNanoTTS.assert_called_once()
    mock_instance.synthesize_to_file.assert_called_with(
        text="Hello World", 
        output_path="output_path.wav",
        voice="expr-voice-4-f",
        speed=1.2
    )
    assert output == "output_path.wav"

@patch("src.voice_cloning.tts.chatterbox.synthesize_with_chatterbox")
@patch("tempfile.mktemp")
def test_generate_speech_chatterbox(mock_mktemp, mock_chatterbox):
    """Test that generating speech with Chatterbox calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    
    output = generate_speech(
        "Chatterbox", "Hello", "ref.wav",
        1.0, False,
        "v", "l", "kv",
        0.8, 0.6, "fr", True,
        0.7, 0.95, True,
        ""
    )
    
    mock_chatterbox.assert_called_with(
        text="Hello",
        output_wav="output_path.wav",
        source_wav="ref.wav",
        exaggeration=0.8,
        cfg_weight=0.6,
        language="fr",
        multilingual=True,
        use_mlx=False
    )
    assert output == "output_path.wav"

@patch("src.voice_cloning.tts.marvis.MarvisTTS")
@patch("tempfile.mktemp")
def test_generate_speech_marvis(mock_mktemp, MockMarvisTTS):
    """Test that generating speech with Marvis calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    mock_instance = MockMarvisTTS.return_value
    
    output = generate_speech(
        "Marvis", "Hello", "ref.wav",
        1.1, True,
        "v", "l", "kv",
        0.7, 0.5, "en", False,
        0.6, 0.9, False,
        ""
    )
    
    mock_instance.synthesize.assert_called_with(
        text="Hello", 
        output_path="output_path.wav", 
        ref_audio="ref.wav",
        speed=1.1,
        temperature=0.6,
        top_p=0.9,
        quantized=False
    )
    assert output == "output_path.wav"

@patch("src.voice_cloning.tts.cosyvoice.synthesize_speech")
@patch("tempfile.mktemp")
def test_generate_speech_cosyvoice(mock_mktemp, mock_cosyvoice):
    """Test that generating speech with CosyVoice calls the correct function."""
    mock_mktemp.return_value = "output_path.wav"
    
    output = generate_speech(
        "CosyVoice", "Hello", "ref.wav",
        1.0, True,
        "v", "l", "kv",
        0.7, 0.5, "en", False,
        0.7, 0.95, True,
        "Excited"
    )
    
    mock_cosyvoice.assert_called_with(
        text="Hello",
        output_path="output_path.wav",
        ref_audio_path="ref.wav",
        instruct_text="Excited",
        speed=1.0,
        use_mlx=True
    )
    assert output == "output_path.wav"

def test_generate_speech_missing_reference():
    """Test that voice cloning models raise error if reference is missing."""
    with pytest.raises(gr.Error, match="Reference audio is required"):
        generate_speech(
            "Chatterbox", "Hello", None,
            1.0, False, "v", "l", "kv", 0.7, 0.5, "en", False, 0.7, 0.95, True, ""
        )

def test_generate_speech_unknown_model():
    """Test that generating speech with an unknown model raises Error."""
    with pytest.raises(gr.Error):
        generate_speech(
            "Unknown", "Hello", None,
            1.0, False, "v", "l", "kv", 0.7, 0.5, "en", False, 0.7, 0.95, True, ""
        )
