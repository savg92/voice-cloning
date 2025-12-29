from typing import Any
from ..base import ModelBenchmark, BenchmarkType

class ParakeetBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Parakeet TDT 0.6B", BenchmarkType.ASR)
        self.model_instance = None

    def load(self):
        from src.voice_cloning.asr.parakeet import ParakeetASR
        self.model_instance = ParakeetASR()
        # Ensure it loaded or we warn
        if self.model_instance.backend == "nemo" and self.model_instance.model is None:
             raise RuntimeError(self.model_instance.err_msg or "Failed to load Parakeet (NeMo)")

    def warmup(self, output_dir: str):
        pass

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        result = self.model_instance.transcribe(input_data)
        
        if result.startswith("Error:"):
             raise RuntimeError(result)

        # Save transcription
        with open(output_path, "w") as f:
            f.write(result)
            
        return {'transcription': result}
