from typing import Tuple, Dict, Any
import numpy as np
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class KittenBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("KittenTTS (Nano)", BenchmarkType.TTS)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.tts.kitten_nano import KittenNanoTTS
        self.model_instance = KittenNanoTTS("KittenML/kitten-tts-nano-0.2")

    def warmup(self, output_dir: str):
        self.model_instance.synthesize_to_file("Warmup", f"{output_dir}/warmup_kitten.wav")

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        self.model_instance.synthesize_to_file(input_data, output_path)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
