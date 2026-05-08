from typing import Any
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class Supertonic3Benchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Supertonic-3", BenchmarkType.TTS)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.tts.supertonic3 import Supertonic3TTS
        self.model_instance = Supertonic3TTS()

    def warmup(self, output_dir: str):
        output_path = f"{output_dir}/warmup_supertonic3.wav"
        self.model_instance.synthesize("Warmup text for Supertonic 3", output_path=output_path, steps=8)

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        # input_data is text
        self.model_instance.synthesize(input_data, output_path=output_path, steps=8)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
