from typing import Any
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class ChatterboxTurboBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Chatterbox Turbo (MLX)", BenchmarkType.TTS)
        self.model_instance = None

    def load(self):
        pass

    def warmup(self, output_dir: str):
        from src.voice_cloning.tts.chatterbox_turbo import synthesize_with_chatterbox_turbo
        output_path = f"{output_dir}/warmup_chatterbox_turbo.wav"
        synthesize_with_chatterbox_turbo("Warmup", output_wav=output_path, use_mlx=True)

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        from src.voice_cloning.tts.chatterbox_turbo import synthesize_with_chatterbox_turbo
        synthesize_with_chatterbox_turbo(input_data, output_wav=output_path, use_mlx=True)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
