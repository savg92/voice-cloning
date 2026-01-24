import os
import sys
import unittest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from voice_cloning.tts.soprano import synthesize_speech

class TestSoprano(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.text = "Soprano is an extremely lightweight text to speech model."

    def test_pytorch_synthesis(self):
        output_path = os.path.join(self.output_dir, "test_soprano_pytorch.wav")
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print("\nTesting Soprano PyTorch synthesis...")
        result = synthesize_speech(self.text, output_path, use_mlx=False, temperature=0.8, top_p=0.9)
        self.assertTrue(os.path.exists(result))
        print(f"✓ PyTorch synthesis successful: {result}")

    def test_mlx_synthesis(self):
        # Only run on macOS/Apple Silicon if MLX is available
        try:
            import mlx.core
        except ImportError:
            self.skipTest("MLX not available on this system")
            
        output_path = os.path.join(self.output_dir, "test_soprano_mlx.wav")
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print("\nTesting Soprano MLX synthesis...")
        result = synthesize_speech(self.text, output_path, use_mlx=True, temperature=0.5, top_p=0.98)
        self.assertTrue(os.path.exists(result))
        print(f"✓ MLX synthesis successful: {result}")

    def test_streaming_synthesis(self):
        output_path = os.path.join(self.output_dir, "test_soprano_stream.wav")
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print("\nTesting Soprano streaming synthesis (Torch)...")
        result = synthesize_speech(
            "This is a test of the streaming synthesis feature in Soprano.",
            output_path,
            use_mlx=False,
            stream=True
        )
        self.assertTrue(os.path.exists(result))
        print(f"✓ Streaming synthesis successful: {result}")

if __name__ == "__main__":
    unittest.main()
