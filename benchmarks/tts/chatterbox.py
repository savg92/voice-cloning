from typing import Tuple, Dict, Any
import numpy as np
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class ChatterboxBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Chatterbox (MLX)", BenchmarkType.TTS)
        self.model_instance = None

    def load(self):
        pass

    def warmup(self, output_dir: str):
        from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
        output_path = f"{output_dir}/warmup_chatterbox.wav"
        synthesize_with_chatterbox("Warmup", output_wav=output_path, use_mlx=True)

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        from src.voice_cloning.tts.chatterbox import synthesize_with_chatterbox
        synthesize_with_chatterbox(input_data, output_wav=output_path, use_mlx=True)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
