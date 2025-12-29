import sys
import os

sys.path.insert(0, os.getcwd())

try:
    from voice_cloning.tts.kitten_nano import KittenNanoTTS, ensure_espeak_compatibility  # noqa: F401
    print("✓ Imported KittenNanoTTS")
    
    ensure_espeak_compatibility()
    print("✓ Ran ensure_espeak_compatibility")
    
    # We won't instantiate the model if kittentts is not installed, 
    # but we can check if the class is available.
    print("✓ KittenNanoTTS class available")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
