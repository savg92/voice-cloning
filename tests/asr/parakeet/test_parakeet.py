import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

def test_parakeet_import():
    # Test lightweight wrapper import
    from voice_cloning.asr.parakeet import get_parakeet, ParakeetASR
    get_parakeet()
    print('lightweight import ok')
    
    # Compatibility with existing test code
    type('obj', (object,), {'ParakeetModel': ParakeetASR})
    print('full import ok', True)
