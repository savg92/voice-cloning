from typing import Any
import soundfile as sf
import shutil
from pathlib import Path
from ..base import ModelBenchmark, BenchmarkType

class MarvisBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("Marvis (MLX)", BenchmarkType.TTS)

    def load(self):
        pass

    def warmup(self, output_dir: str):
        pass

    def run_test(self, input_data: Any, output_path: str) -> dict[str, Any]:
        from src.voice_cloning.tts.marvis import MarvisTTS
        model = MarvisTTS()
        
        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # We need to guarantee a clean slate because Marvis wrapper uses specific naming 
        if output_path_obj.exists():
            output_path_obj.unlink()
            
        # The Marvis wrapper inside 'src' uses a temporary directory logic but copies the final file 
        # to the 'output_path' we pass it. 
        # However, the previous error "Marvis output file not found" suggests the rename/move failed 
        # or the file wasn't generated.
        
        # Let's try calling it.
        try:
            model.synthesize(input_data, output_path=output_path)
        except Exception as e:
            raise RuntimeError(f"Marvis synthesis call failed: {e}")
        
        # Check if file exists now
        if not output_path_obj.exists():
             # Last ditch effort: output might be named with _000 suffix if logic inside Marvis wrapper is slightly off
             # regarding the destination path.
             possible_suffix = output_path_obj.parent / (output_path_obj.stem + "_000.wav")
             if possible_suffix.exists():
                 shutil.move(str(possible_suffix), output_path)
             else:
                 raise FileNotFoundError(f"Marvis output not found at {output_path} (or {possible_suffix})")
            
        audio, sr = sf.read(output_path)
        return {'audio': audio, 'sr': sr}
