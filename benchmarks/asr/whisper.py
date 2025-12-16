from typing import Tuple, Dict, Any
import numpy as np
import time
from ..base import ModelBenchmark, BenchmarkType

class WhisperBenchmark(ModelBenchmark):
    def __init__(self, use_mlx: bool = True, model_id: str = "mlx-community/whisper-large-v3-turbo"):
        # Simplify name for display
        friendly_name = model_id.split("/")[-1].replace("whisper-", "")
        name = f"Whisper {friendly_name} ({'MLX' if use_mlx else 'PyTorch'})"
        
        super().__init__(name, BenchmarkType.ASR)
        self.use_mlx = use_mlx
        self.model_id = model_id
        self.model_instance = None

    def load(self):
        from src.voice_cloning.asr.whisper import WhisperASR
        self.model_instance = WhisperASR(model_id=self.model_id, use_mlx=self.use_mlx)
        self.model_instance.load_model()

    def warmup(self, output_dir: str):
        pass 

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        result = self.model_instance.transcribe(input_data)
        
        # Save transcription
        with open(output_path, "w") as f:
            f.write(result)
            
        return {'transcription': result}
