from typing import Tuple, Dict, Any
import numpy as np
import json
from ..base import ModelBenchmark, BenchmarkType

class HumAwareBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("HumAware VAD", BenchmarkType.VAD)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.vad.humaware import HumAwareVAD
        self.model_instance = HumAwareVAD()

    def warmup(self, output_dir: str):
        pass

    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        timestamp_list = self.model_instance.detect_speech(input_data)
        
        # Save result
        with open(output_path, "w") as f:
            json.dump(timestamp_list, f, indent=2)
            
        return {'segments': timestamp_list}
