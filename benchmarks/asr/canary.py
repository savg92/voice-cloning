from typing import Tuple, Dict, Any
import numpy as np
from ..base import ModelBenchmark, BenchmarkType

class CanaryBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Canary-1B-v2", BenchmarkType.ASR)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.asr.canary import CanaryASR
        self.model_instance = CanaryASR()
        if not self.model_instance.load_model():
             raise RuntimeError("Failed to load Canary model")

    def warmup(self, output_dir: str):
        pass

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        result = self.model_instance.transcribe(input_data)
        
        # Save transcription
        with open(output_path, "w") as f:
            f.write(result['text'])
            
        return {'transcription': result['text']}
