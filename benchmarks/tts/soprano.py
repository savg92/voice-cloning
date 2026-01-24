from typing import Any
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class SopranoBenchmark(ModelBenchmark):
    def __init__(self, use_mlx: bool = False):
        name = "Soprano (MLX)" if use_mlx else "Soprano"
        super().__init__(name, BenchmarkType.TTS)
        self.use_mlx = use_mlx

    def load(self):
        # Imports are done inside methods to avoid slow startup and dependency issues
        if self.use_mlx:
            from mlx_audio.tts.utils import load_model
            self.model = load_model("mlx-community/Soprano-80M-6bit")
        else:
            from soprano import SopranoTTS
            self.model = SopranoTTS(backend='auto', device='auto')

    def warmup(self, output_dir: str):
        from src.voice_cloning.tts.soprano import synthesize_speech
        synthesize_speech("Warmup", output_path=f"{output_dir}/warmup_soprano_{'mlx' if self.use_mlx else 'torch'}.wav", use_mlx=self.use_mlx)

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        from src.voice_cloning.tts.soprano import synthesize_speech
        
        # Use stable defaults for benchmarking
        synthesize_speech(
            input_data, 
            output_path=output_path, 
            use_mlx=self.use_mlx,
            temperature=0.7,
            top_p=0.95
        )
        
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
