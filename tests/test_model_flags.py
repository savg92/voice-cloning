import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
from src.voice_cloning.tts.marvis import MarvisTTS

def test_chatterbox_mlx_flags():
    """Test that flags are passed correctly to the MLX backend in Chatterbox."""
    with patch("src.voice_cloning.tts.chatterbox._synthesize_with_mlx") as mock_mlx:
        synthesize_with_chatterbox(
            text="Hello",
            output_wav="out.wav",
            use_mlx=True,
            speed=1.5,
            stream=True,
            language="es",
            voice="test_voice",
            exaggeration=0.8,
            cfg_weight=0.7
        )
        
        mock_mlx.assert_called_once()
        args, kwargs = mock_mlx.call_args
        
        # Check kwargs for named arguments
        assert kwargs["speed"] == 1.5
        assert kwargs["stream"] is True
        assert kwargs["voice"] == "test_voice"
        
        # Check positional arguments
        # Signature: text, output_wav, source_wav, exaggeration, cfg_weight, language
        assert args[0] == "Hello"
        assert args[3] == 0.8  # exaggeration
        assert args[4] == 0.7  # cfg_weight
        assert args[5] == "es" # language

def test_marvis_subprocess_flags():
    """Test that MarvisTTS constructs the subprocess command with correct flags."""
    marvis = MarvisTTS()
    
    with patch("subprocess.run") as mock_run:
        # Mock successful execution
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        
        # Mock shutil.move to avoid file error since we don't actually generate files
        with patch("shutil.move"), \
             patch("pathlib.Path.exists", return_value=True):
            
            marvis.synthesize(
                text="Hello",
                output_path="out.wav",
                speed=1.2,
                stream=True,
                lang_code="fr",
                voice="marvis_voice",
                temperature=0.9
            )
            
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            cmd = args[0]
            
            # Verify flags in command list
            assert "--speed" in cmd
            assert "1.2" in cmd
            assert "--stream" in cmd
            assert "--lang_code" in cmd
            assert "f" in cmd # 'fr' maps to 'f'
            assert "--voice" in cmd
            assert "marvis_voice" in cmd
            assert "--temperature" in cmd
            assert "0.9" in cmd
