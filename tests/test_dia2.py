"""
Tests for Dia2-1B TTS model wrapper.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestDia2TTS:
    """Test suite for Dia2TTS wrapper."""

    @pytest.fixture
    def mock_dia2_imports(self):
        """Mock the dia2 library imports."""
        with patch("src.voice_cloning.tts.dia2.torch") as mock_torch, \
             patch("src.voice_cloning.tts.dia2.logger"):
            
            # Setup torch mocks
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            
            yield mock_torch

    @pytest.fixture
    def mock_dia2_model(self):
        """Mock the Dia2 model and related classes."""
        # Patch the import location
        with patch.dict('sys.modules', {'dia2': MagicMock()}):
            import sys
            mock_dia2_module = sys.modules['dia2']
            
            # Create mock classes
            MockDia2 = Mock()
            MockGenConfig = Mock()
            MockSampConfig = Mock()
            
            mock_dia2_module.Dia2 = MockDia2
            mock_dia2_module.GenerationConfig = MockGenConfig
            mock_dia2_module.SamplingConfig = MockSampConfig
            
            # Create mock model instance
            mock_model = Mock()
            MockDia2.from_repo.return_value = mock_model
            
            # Create mock result
            mock_result = Mock()
            mock_waveform = np.random.randn(16000)  # 1 second at 16kHz
            mock_result.waveform = mock_waveform
            mock_model.generate.return_value = mock_result
            
            yield {
                "Dia2": MockDia2,
                "GenerationConfig": MockGenConfig,
                "SamplingConfig": MockSampConfig,
                "model": mock_model,
                "result": mock_result,
                "waveform": mock_waveform
            }

    def test_model_initialization(self, mock_dia2_imports, mock_dia2_model):
        """Test that the model initializes correctly."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        Dia2TTS(device="cpu")
        
        # Verify from_repo was called
        mock_dia2_model["Dia2"].from_repo.assert_called_once_with(
            "nari-labs/Dia2-1B",
            device="cpu",
            dtype="bfloat16"
        )

    def test_device_auto_detection_cpu(self, mock_dia2_imports, mock_dia2_model):
        """Test automatic device selection falls back to CPU."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        # Ensure CUDA and MPS are not available
        mock_dia2_imports.cuda.is_available.return_value = False
        
        tts = Dia2TTS()
        
        assert tts.device == "cpu"

    def test_device_auto_detection_cuda(self, mock_dia2_imports, mock_dia2_model):
        """Test automatic device selection prefers CUDA."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        mock_dia2_imports.cuda.is_available.return_value = True
        
        tts = Dia2TTS()
        
        assert tts.device == "cuda"

    def test_synthesize_basic(self, mock_dia2_imports, mock_dia2_model):
        """Test basic synthesis."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        tts = Dia2TTS(device="cpu")
        audio = tts.synthesize(text="[S1] Hello world!")
        
        # Verify generate was called
        mock_dia2_model["model"].generate.assert_called_once()
        
        # Verify audio is numpy array
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1

    def test_synthesize_with_parameters(self, mock_dia2_imports, mock_dia2_model):
        """Test synthesis with custom parameters."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        tts = Dia2TTS(device="cpu")
        tts.synthesize(
            text="[S1] Test",
            cfg_scale=3.0,
            temperature=0.9,
            top_k=100,
            use_cuda_graph=False,
            verbose=True
        )
        
        # Verify  GenerationConfig and SamplingConfig were created with correct params
        mock_dia2_model["SamplingConfig"].assert_called_with(temperature=0.9, top_k=100)
        mock_dia2_model["GenerationConfig"].assert_called()

    def test_synthesize_with_output_path(self, mock_dia2_imports, mock_dia2_model):
        """Test synthesis with output file."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        tts = Dia2TTS(device="cpu")
        output_path = "test.wav"
        
        tts.synthesize(
            text="[S1] Test",
            output_path=output_path
        )
        
        # Verify generate was called with output_wav
        call_kwargs = mock_dia2_model["model"].generate.call_args[1]
        assert "output_wav" in call_kwargs
        assert call_kwargs["output_wav"] == output_path

    def test_synthesize_with_voice_cloning(self, mock_dia2_imports, mock_dia2_model):
        """Test synthesis with voice cloning."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        tts = Dia2TTS(device="cpu")
        
        tts.synthesize(
            text="[S1] Test",
            prefix_speaker_1="voice1.wav",
            prefix_speaker_2="voice2.wav"
        )
        
        # Verify generate was called with prefix audio
        call_kwargs = mock_dia2_model["model"].generate.call_args[1]
        assert "prefix_speaker_1" in call_kwargs
        assert call_kwargs["prefix_speaker_1"] == "voice1.wav"
        assert "prefix_speaker_2" in call_kwargs
        assert call_kwargs["prefix_speaker_2"] == "voice2.wav"


    def test_repr(self, mock_dia2_imports, mock_dia2_model):
        """Test string representation."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        tts = Dia2TTS(model_name="nari-labs/Dia2-2B", device="cuda", dtype="float16")
        
        repr_str = repr(tts)
        assert "Dia2TTS" in repr_str
        assert "nari-labs/Dia2-2B" in repr_str
        assert "cuda" in repr_str
        assert "float16" in repr_str

    def test_waveform_tensor_conversion(self, mock_dia2_imports, mock_dia2_model):
        """Test that torch tensor waveforms are converted to numpy."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        import torch
        
        # Mock waveform as torch tensor
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(16000)
        mock_dia2_model["result"].waveform = mock_tensor
        
        tts = Dia2TTS(device="cpu")
        audio = tts.synthesize(text="[S1] Test")
        
        # Verify tensor was converted
        mock_tensor.cpu.assert_called_once()
        mock_tensor.cpu.return_value.numpy.assert_called_once()
        assert isinstance(audio, np.ndarray)

    def test_multidimensional_waveform_squeeze(self, mock_dia2_imports, mock_dia2_model):
        """Test that multi-dimensional waveforms are squeezed to 1D."""
        from src.voice_cloning.tts.dia2 import Dia2TTS
        
        # Mock waveform as 2D array
        mock_dia2_model["result"].waveform = np.random.randn(1, 16000)
        
        tts = Dia2TTS(device="cpu")
        audio = tts.synthesize(text="[S1] Test")
        
        # Verify output is 1D
        assert audio.ndim == 1
        assert audio.shape == (16000,)
