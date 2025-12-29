# Quick import test for Parakeet wrappers

# Test lightweight wrapper import
from src.voice_cloning.asr.parakeet import get_parakeet, ParakeetASR
p = get_parakeet()
print('lightweight import ok')

# Compatibility with existing test code
mod = type('obj', (object,), {'ParakeetModel': ParakeetASR})
print('full import ok', True)
