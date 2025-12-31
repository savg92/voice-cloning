
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chatterbox_loading():
    print("Testing Chatterbox monkeypatch...")
    
    try:
        from voice_cloning.tts.chatterbox import _synthesize_with_mlx
        from mlx_audio.tts.models.chatterbox.chatterbox import Model as ChatterboxModel
        
        print("Verifying if we can import the necessary modules...")
        import mlx_audio
        print(f"mlx_audio version: {mlx_audio.__version__}")
        print(f"Chatterbox class imported successfully as: {ChatterboxModel}")
        
    except ImportError as e:
        print(f"Import failed: {e}")
        return

    print("Import successful. Verification script finished.")

if __name__ == "__main__":
    test_chatterbox_loading()
