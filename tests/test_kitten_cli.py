import sys
import types
from pathlib import Path
import pytest


def test_kitten_cli_writes_file(monkeypatch, tmp_path):
    # Inject fake kittentts
    fake_kittentts = types.ModuleType('kittentts')
    class DummyKittenTTS:
        def __init__(self, model_id, cache_dir=None):
            self.model_id = model_id
        def generate(self, text, voice='expr-voice-4-f', speed=1.0):
            return [0.0, 0.1, -0.1, 0.0]
    fake_kittentts.KittenTTS = DummyKittenTTS
    monkeypatch.setitem(sys.modules, 'kittentts', fake_kittentts)

    # Inject fake numpy
    fake_np = types.ModuleType('numpy')
    def asarray(x, dtype=None):
        return list(x)
    fake_np.asarray = asarray
    monkeypatch.setitem(sys.modules, 'numpy', fake_np)

    # Inject fake soundfile
    fake_sf = types.ModuleType('soundfile')
    def write(path, arr, sr):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(' '.join(str(x) for x in arr))
    fake_sf.write = write
    monkeypatch.setitem(sys.modules, 'soundfile', fake_sf)

    # Prepare the CLI args
    out_path = tmp_path / 'cli_out.wav'
    monkeypatch.setattr(sys, 'argv', ['scripts/kitten_cli.py', '--text', 'hello test', '--output', str(out_path)])

    # Call the CLI main function
    from scripts.kitten_cli import main
    main()

    assert out_path.exists(), 'CLI should write output file'
