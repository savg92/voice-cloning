from typing import Any
import soundfile as sf
from ..base import ModelBenchmark, BenchmarkType
import torch

class NeuTTSBenchmark(ModelBenchmark):
    def __init__(self, ref_audio_path: str = None, ref_text_path: str = None):
        super().__init__("NeuTTS Air", BenchmarkType.TTS)
        self.model_instance = None
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path

    def load(self):
        from src.voice_cloning.tts.neutts_air import NeuTTSAirTTS
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_instance = NeuTTSAirTTS(backbone_device=device, codec_device=device)

    def warmup(self, output_dir: str):
        if not self.ref_audio_path:
             return

        self.model_instance.synthesize(
            text="Warmup",
            output_path=f"{output_dir}/warmup_neutts.wav",
            ref_audio_path=self.ref_audio_path,
            ref_text_path=self.ref_text_path
        )

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        if not self.ref_audio_path:
             raise ValueError("NeuTTS requires reference audio")

        self.model_instance.synthesize(
            text=input_data,
            output_path=output_path,
            ref_audio_path=self.ref_audio_path,
            ref_text_path=self.ref_text_path
        )
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
