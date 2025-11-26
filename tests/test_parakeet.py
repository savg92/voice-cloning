# Quick import test for Parakeet wrappers
from pathlib import Path
import importlib.util

# Test lightweight wrapper import
from src.voice_cloning.asr.parakeet import get_parakeet
p = get_parakeet()
print('lightweight import ok')

# Load full implementation via importlib (filename contains hyphens)
spec = importlib.util.spec_from_file_location('parakeet_v3', Path('src/voice_cloning/asr/parakeet_tdt_0_6b_v3.py').resolve())
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('full import ok', hasattr(mod, 'ParakeetModel'))
