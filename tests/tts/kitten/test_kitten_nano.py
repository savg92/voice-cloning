import sys
from pathlib import Path
import types
import numpy as np
import soundfile as sf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from voice_cloning.tts.kitten_nano import KittenNanoTTS


def test_kitten_nano_synthesize_file(tmp_path, monkeypatch):
    """Test that KittenNanoTTS wraps and saves audio correctly.

    We patch the 'kittentts' module to avoid network or large model download
    during testing.
    """
    # Create a fake kittentts module
    fake = types.ModuleType("kittentts")

    class DummyKittenTTS:
        def __init__(self, model_id, cache_dir=None):
            self.model_id = model_id

        def generate(self, text, voice="expr-voice-4-f", speed=1.0):
            # Return a short 0.05s sine wave at 24000Hz to ensure non-empty audio
            sr = 24000
            t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False, dtype=np.float32)
            wav = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
            return wav

    fake.KittenTTS = DummyKittenTTS
    monkeypatch.setitem(sys.modules, "kittentts", fake)

    # Now instantiate the wrapper and synthesize
    tts = KittenNanoTTS()
    out_file = tmp_path / "kitten_test.wav"
    tts.synthesize_to_file("Hello from tests", out_file)

    assert out_file.exists(), "Expected synthesized file to exist"
    info = sf.info(str(out_file))
    assert info.samplerate == 24000
    assert info.frames > 0
