
from src.voice_cloning.asr.parakeet import ParakeetASR


def test_parakeet_cli_missing(monkeypatch, tmp_path):
    """When MLX is selected but CLI helper is missing, ensure an error is returned and helpful guidance.
    """
    # Simulate 'uv' and 'parakeet-mlx' missing from PATH before model initialization
    monkeypatch.setattr("src.voice_cloning.asr.parakeet.shutil.which", lambda name: None)

    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")

    model = ParakeetASR()

    transcript = model.transcribe(str(audio), timestamps=False)
    assert transcript.startswith("Error:"), f"Expected Error output when binaries missing, got: {transcript}"
    assert "MLX parakeet CLI not found" in transcript or "MLX runner missing" in transcript


def test_parakeet_nemo_missing(monkeypatch, tmp_path):
    """When the NeMo toolkit isn't installed and MLX isn't available, ensure the transcribe method returns an Error with guidance."""
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"")

    model = ParakeetASR()
    # Force to Nemo backend to exercise the NeMo path
    model.backend = "nemo"

    # Ensure the model appears not to be loaded
    model.model = None
    model.err_msg = "NeMo toolkit not installed (simulated)"

    transcript = model.transcribe(str(audio), timestamps=False)
    assert transcript.startswith("Error:"), f"Expected Error output when NeMo missing, got: {transcript}"
    assert "NeMo toolkit not installed" in transcript
