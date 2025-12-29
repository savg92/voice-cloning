import sys
import types
import pytest


def test_kitten_cli_espeak_attribute_error(monkeypatch, tmp_path, capsys):
    # Create a fake kittentts that raises AttributeError simulating a phonemizer/espeak mismatch
    fake_kittentts = types.ModuleType('kittentts')
    class BrokenKittenTTS:
        def __init__(self, model_id, cache_dir=None):
            raise AttributeError("type object 'EspeakWrapper' has no attribute 'set_data_path'")
    fake_kittentts.KittenTTS = BrokenKittenTTS
    monkeypatch.setitem(sys.modules, 'kittentts', fake_kittentts)

    # Fake numpy and soundfile
    fake_np = types.ModuleType('numpy')
    fake_np.asarray = lambda x, dtype=None: list(x)
    monkeypatch.setitem(sys.modules, 'numpy', fake_np)

    fake_sf = types.ModuleType('soundfile')
    def write(path, arr, sr): pass
    fake_sf.write = write
    monkeypatch.setitem(sys.modules, 'soundfile', fake_sf)

    # Run the CLI and capture exit
    monkeypatch.setattr(sys, 'argv', ['scripts/kitten_cli.py', '--text', 'Hello', '--output', str(tmp_path / 'o.wav')])
    from scripts.kitten_cli import main

    with pytest.raises(SystemExit) as se:
        main()
    # exit code should be 1 on failure
    assert se.value.code == 1
