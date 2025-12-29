import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

def test_chatterbox():
    print("Testing Chatterbox integration...")
    try:
        from voice_cloning.tts.chatterbox import ChatterboxWrapper
        wrapper = ChatterboxWrapper()
        print("✓ ChatterboxWrapper initialized successfully")
        
        # Check if model has generate method
        if hasattr(wrapper.model, "generate"):
            print("✓ Model has generate method")
        else:
            print("✗ Model missing generate method")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_chatterbox()
