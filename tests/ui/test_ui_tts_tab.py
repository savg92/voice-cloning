import pytest
import gradio as gr
from unittest.mock import patch
from voice_cloning.ui.tts_tab import generate_speech

def get_default_args():
    return [
        "Kokoro", "Text", None, None, 1.0, True, False, # model, text, ref, ref_text, speed, mlx, stream
        "a", "af_heart", # kokoro (lang then voice)
        "v", # kitten
        0.7, 0.5, "en", False, "", "", # chatter
        0.7, 0.95, True, # marvis
        "", # cosy
        "backbone", # neutts
        "preset", 8, 1.0, # supertone
        2.0, 0.8, 50 # dia2
    ]

@patch("voice_cloning.tts.kokoro.synthesize_speech")
@patch("tempfile.mktemp")
def test_generate_speech_kokoro(mock_mktemp, mock_kokoro):
    mock_mktemp.return_value = "output.wav"
    args = get_default_args()
    args[0] = "Kokoro"
    args[1] = "Hello"
    
    generate_speech(*args)
    
    mock_kokoro.assert_called_with(
        text="Hello", output_path="output.wav", voice="af_heart",
        lang_code="a", speed=1.0, use_mlx=True, stream=False
    )

@patch("voice_cloning.tts.supertone.synthesize_with_supertone")
@patch("tempfile.mktemp")
def test_generate_speech_supertone(mock_mktemp, mock_supertone):
    mock_mktemp.return_value = "output.wav"
    args = get_default_args()
    args[0] = "Supertone"
    args[21] = "my-preset"
    
    generate_speech(*args)
    
    mock_supertone.assert_called_with(
        text="Text", output_path="output.wav", preset="my-preset",
        steps=8, cfg_scale=1.0, stream=False
    )

def test_generate_speech_missing_text():
    args = get_default_args()
    args[1] = ""
    with pytest.raises(gr.Error, match="Please enter some text"):
        generate_speech(*args)
