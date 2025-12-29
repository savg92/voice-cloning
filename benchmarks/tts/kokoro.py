from typing import Any
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType

class KokoroBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Kokoro", BenchmarkType.TTS)

    def load(self):
        pass

    def warmup(self, output_dir: str):
        from src.voice_cloning.tts.kokoro import synthesize_speech
        synthesize_speech("Warmup", output_path=f"{output_dir}/warmup_kokoro.wav")

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        from src.voice_cloning.tts.kokoro import synthesize_speech
        synthesize_speech(input_data, output_path=output_path)
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
