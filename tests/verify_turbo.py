import sys
import unittest
from unittest.mock import patch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox

class TestChatterboxTurbo(unittest.TestCase):
    
    @patch('src.voice_cloning.tts.chatterbox._synthesize_with_mlx')
    def test_model_id_passed_to_mlx(self, mock_mlx):
        """Test that model_id is correctly passed down to the MLX synthesis function."""
        text = "Hello"
        output = "out.wav"
        model_id = "mlx-community/chatterbox-turbo-4bit"
        
        synthesize_with_chatterbox(
            text=text,
            output_wav=output,
            use_mlx=True,
            model_id=model_id
        )
        
        # Verify it was called with the correct model_id
        mock_mlx.assert_called_once()
        args, kwargs = mock_mlx.call_args
        self.assertEqual(kwargs.get('model_id'), model_id)
        print("✓ model_id correctly passed to _synthesize_with_mlx")

    @patch('src.voice_cloning.tts.chatterbox._synthesize_with_mlx')
    def test_default_model_id(self, mock_mlx):
        """Test that it defaults to None (which then defaults to 4bit) if not provided."""
        synthesize_with_chatterbox(
            text="Hello",
            output_wav="out.wav",
            use_mlx=True
        )
        
        mock_mlx.assert_called_once()
        args, kwargs = mock_mlx.call_args
        self.assertIsNone(kwargs.get('model_id'))
        print("✓ Default model_id (None) correctly passed")

if __name__ == "__main__":
    unittest.main()
