import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    print("Testing imports...")
    try:
        from src.voice_cloning.vad.humaware import HumAwareVAD  # noqa: F401
        print("✓ HumAwareVAD imported")
    except ImportError as e:
        print(f"✗ HumAwareVAD import failed: {e}")

    try:
        from src.voice_cloning.asr.parakeet import ParakeetASR  # noqa: F401
        print("✓ ParakeetASR imported")
    except ImportError as e:
        print(f"✗ ParakeetASR import failed: {e}")

    try:
        from src.voice_cloning.tts.marvis import MarvisTTS  # noqa: F401
        print("✓ MarvisTTS imported")
    except ImportError as e:
        print(f"✗ MarvisTTS import failed: {e}")

    try:
        from src.voice_cloning.tts.maya import Maya1  # noqa: F401
        print("✓ Maya1 imported")
    except ImportError as e:
        print(f"✗ Maya1 import failed: {e}")

    try:
        from src.voice_cloning.tts.kitten_nano import KittenNanoTTS  # noqa: F401
        print("✓ KittenNanoTTS imported")
    except ImportError as e:
        print(f"✗ KittenNanoTTS import failed: {e}")

if __name__ == "__main__":
    test_imports()
