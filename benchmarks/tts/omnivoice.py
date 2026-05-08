from typing import Any
import soundfile as sf
import torch
from ..base import ModelBenchmark, BenchmarkType

class OmniVoiceBenchmark(ModelBenchmark):
    def __init__(self, device: str = None):
        super().__init__("OmniVoice", BenchmarkType.TTS)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_instance = None

    def load(self):
        from src.voice_cloning.tts.omnivoice import OmniVoiceTTS
        self.model_instance = OmniVoiceTTS(device=self.device)

    def warmup(self, output_dir: str):
        output_path = f"{output_dir}/warmup_omnivoice.wav"
        self.model_instance.synthesize("Warmup", output_path=output_path)

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        self.model_instance.synthesize(input_data, output_path=output_path)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
