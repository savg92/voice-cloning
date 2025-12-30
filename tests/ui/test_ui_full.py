import pytest
import gradio as gr
from unittest.mock import patch
import os

# UI components use local imports within functions. 
# We target the module exactly as it is imported in the code being tested.

from voice_cloning.ui.app import create_interface
from voice_cloning.ui.asr_tab import create_asr_tab, transcribe_speech
from voice_cloning.ui.tts_tab import generate_speech
from voice_cloning.ui.vad_tab import create_vad_tab, detect_speech_segments

class TestUIFull:
    """Consolidated UI test suite."""

    @pytest.fixture
    def real_sample_wav(self):
        path = "samples/anger.wav"
        if not os.path.exists(path):
            # Create a valid minimal wav file if missing
            import soundfile as sf
            import numpy as np
            os.makedirs("samples", exist_ok=True)
            sf.write(path, np.zeros(16000), 16000)
        return path

    # 1. App Level Tests
    def test_create_interface(self):
        """Test that the main interface creates correctly."""
        demo = create_interface()
        assert isinstance(demo, gr.Blocks)
        assert "Voice Cloning Toolkit" in demo.title

    # 2. ASR Tab Tests
    def test_create_asr_tab(self):
        """Test ASR tab creation."""
        with gr.Blocks():
            tab = create_asr_tab()
            assert isinstance(tab, (gr.blocks.BlockContext, gr.components.Component))

    @patch("src.voice_cloning.asr.whisper.WhisperASR")
    def test_transcribe_speech_whisper(self, MockWhisper, real_sample_wav):
        """Test Whisper integration in ASR tab."""
        mock_instance = MockWhisper.return_value
        mock_instance.transcribe.return_value = "Whisper transcript"
        
        output = transcribe_speech(
            "Whisper", real_sample_wav, 
            "openai/whisper-tiny", "en", "transcribe", False, True,
            False, "en", "en", False
        )
        
        assert output == "Whisper transcript"

    @patch("src.voice_cloning.asr.parakeet.ParakeetASR")
    def test_transcribe_speech_parakeet(self, MockParakeet, real_sample_wav):
        """Test Parakeet integration in ASR tab."""
        mock_instance = MockParakeet.return_value
        mock_instance.transcribe.return_value = "Parakeet transcript"
        
        output = transcribe_speech(
            "Parakeet", real_sample_wav,
            "", "", "", False, False, False, "", "", False
        )
        
        assert output == "Parakeet transcript"

    # 3. TTS Tab Tests
    def get_default_tts_args(self):
        # Must match generate_speech signature exactly: 28 arguments.
        return [
            "Kokoro", "Text", None, None, 1.0, True, False, # basic (7)
            "a", "af_heart", # kokoro (2)
            "v0.2", "expr-voice-4-f", # kitten (2)
            0.7, 0.5, "en", False, "", "", # chatter (6)
            0.7, 0.95, True, # marvis (3)
            "", # cosy (1)
            "backbone", # neutts (1)
            "preset", 8, 1.0, # supertone (3)
            2.0, 0.8, 50 # dia2 (3)
        ]

    @patch("src.voice_cloning.tts.kokoro.synthesize_speech")
    @patch("tempfile.mktemp")
    def test_generate_speech_kokoro(self, mock_mktemp, mock_kokoro):
        mock_mktemp.return_value = "output.wav"
        args = self.get_default_tts_args()
        args[0] = "Kokoro"
        args[1] = "Hello"
        
        generate_speech(*args)
        mock_kokoro.assert_called()

    @patch("src.voice_cloning.tts.marvis.MarvisTTS")
    @patch("tempfile.mktemp")
    def test_generate_speech_marvis(self, mock_mktemp, MockMarvis, real_sample_wav):
        mock_mktemp.return_value = "output.wav"
        mock_instance = MockMarvis.return_value
        args = self.get_default_tts_args()
        args[0] = "Marvis"
        args[1] = "Hello Marvis"
        args[2] = real_sample_wav
        
        generate_speech(*args)
        mock_instance.synthesize.assert_called()

    @patch("src.voice_cloning.tts.supertone.synthesize_with_supertone")
    @patch("tempfile.mktemp")
    def test_generate_speech_supertone(self, mock_mktemp, mock_supertone):
        mock_mktemp.return_value = "output.wav"
        args = self.get_default_tts_args()
        args[0] = "Supertone"
        args[22] = "M1"
        
        generate_speech(*args)
        mock_supertone.assert_called()

    # 4. VAD Tab Tests
    def test_create_vad_tab(self):
        """Test VAD tab creation."""
        with gr.Blocks():
            tab = create_vad_tab()
            assert isinstance(tab, (gr.blocks.BlockContext, gr.components.Component))

    @patch("voice_cloning.ui.vad_tab.HumAwareVAD")
    def test_detect_speech_segments(self, MockVAD, real_sample_wav):
        """Test VAD analysis integration."""
        mock_instance = MockVAD.return_value
        mock_instance.detect_speech.return_value = [{'start': 0.0, 'end': 1.0}]
        
        output = detect_speech_segments(real_sample_wav, 0.5, 250, 100, 30)
        
        MockVAD.assert_called_once()
        assert "0.0" in output
        assert "1.0" in output

    def test_missing_inputs(self):
        """Test error handling for missing inputs."""
        with pytest.raises(gr.Error, match="Please upload an audio file"):
            transcribe_speech("Whisper", None, "", "", "", False, False, False, "", "", False)
            
        args = self.get_default_tts_args()
        args[1] = "" # Empty text
        with pytest.raises(gr.Error, match="Please enter some text"):
            generate_speech(*args)