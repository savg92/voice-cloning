from typing import Tuple, Dict, Any
import numpy as np
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class SupertoneBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Supertone", BenchmarkType.TTS)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.tts.supertone import SupertoneTTS
        self.model_instance = SupertoneTTS()

    def warmup(self, output_dir: str):
        output_path = f"{output_dir}/warmup_supertone.wav"
        self.model_instance.synthesize("Warmup", output_path=output_path)

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        self.model_instance.synthesize(input_data, output_path=output_path)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
