from typing import Any
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class Supertonic2Benchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Supertonic-2", BenchmarkType.TTS)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.tts.supertonic2 import Supertonic2TTS
        # Hybrid download happens in init, so this covers it
        # Explicitly checking available providers (e.g. if we want to enforce CPU or test CUDA later)
        # But for now, default is fine.
        self.model_instance = Supertonic2TTS()

    def warmup(self, output_dir: str):
        output_path = f"{output_dir}/warmup_supertonic2.wav"
        # Using default settings for warmup
        self.model_instance.synthesize("Warmup text", output_path=output_path, speed=1.0, steps=10)

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        # input_data is text
        self.model_instance.synthesize(input_data, output_path=output_path, speed=1.0, steps=10)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
