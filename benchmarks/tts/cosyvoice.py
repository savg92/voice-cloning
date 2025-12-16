from typing import Tuple, Dict, Any
import numpy as np
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType
import platform

class CosyVoiceBenchmark(ModelBenchmark):
    def __init__(self, use_mlx: bool = True):
        super().__init__("CosyVoice2 (MLX)" if use_mlx else "CosyVoice2 (PyTorch)", BenchmarkType.TTS)
        self.use_mlx = use_mlx
        self.wrapper = None

    def load(self):
        pass

    def warmup(self, output_dir: str):
        from src.voice_cloning.tts.cosyvoice import synthesize_speech
        output_path = f"{output_dir}/warmup_cosyvoice_{'mlx' if self.use_mlx else 'torch'}.wav"
        synthesize_speech(
            "Warmup",
            output_path=output_path,
            use_mlx=self.use_mlx
        )

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        from src.voice_cloning.tts.cosyvoice import synthesize_speech
        synthesize_speech(
            input_data,
            output_path=output_path,
            use_mlx=self.use_mlx
        )
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
