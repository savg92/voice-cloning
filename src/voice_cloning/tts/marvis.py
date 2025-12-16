import torch
import logging
import sys
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class MarvisTTS:
    """
    Wrapper for Marvis-AI/marvis-tts-250m-v0.2.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.backend = "mlx" # Force MLX
        self.model = None
        # No need to load model in python if using subprocess CLI
        
    def _determine_backend(self) -> str:
        return "mlx"

    def _load_model(self):
        # No-op for MLX subprocess approach
        pass

    def synthesize(self, text: str, output_path: str, ref_audio: Optional[str] = None, 
                   ref_text: Optional[str] = None, stream: bool = False, 
                   speed: Optional[float] = None, temperature: Optional[float] = None,
                   quantized: bool = True):
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            ref_audio: Path to reference audio for voice cloning
            ref_text: Optional text caption for reference audio (auto-transcribed if not provided)
            stream: Enable streaming output (audio plays as it generates)
            speed: Speech speed multiplier (default: 1.0)
            temperature: Sampling temperature for generation (default: 0.7)
        """
        # Using MLX via subprocess as verified
        try:
            import subprocess
            import tempfile
            import shutil
            from pathlib import Path

            # Use a temporary directory to handle the _000.wav suffix
            with tempfile.TemporaryDirectory() as temp_dir:
                # Construct command
                temp_prefix = Path(temp_dir) / "output"
                
                # Check for local quantized model if requested
                local_model_path = Path("models/marvis-4bit")
                model_arg = "Marvis-AI/marvis-tts-250m-v0.2"
                
                if quantized and local_model_path.exists():
                    logger.info(f"Using local quantized model: {local_model_path}")
                    model_arg = str(local_model_path)
                elif quantized and not local_model_path.exists():
                    logger.warning(f"Quantized model not found at {local_model_path}, falling back to standard model")
                
                cmd = [
                    "uv", "run", "python", "-m", "mlx_audio.tts.generate",
                    "--model", model_arg,
                    "--text", text,
                    "--file_prefix", str(temp_prefix)
                ]
                
                # Add voice cloning parameters
                if ref_audio:
                    cmd.extend(["--ref_audio", ref_audio])
                    if ref_text:
                        cmd.extend(["--ref_text", ref_text])
                
                # Add streaming
                if stream:
                    cmd.append("--stream")
                
                # Add speed control
                if speed is not None:
                    cmd.extend(["--speed", str(speed)])
                
                # Add temperature control
                if temperature is not None:
                    cmd.extend(["--temperature", str(temperature)])
                
                logger.info(f"Running Marvis MLX: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Always log stderr to debug issues
                if result.stderr:
                    logger.info(f"Marvis stderr: {result.stderr}")
                if result.stdout:
                    logger.info(f"Marvis stdout: {result.stdout}")
                
                if result.returncode != 0:
                    logger.error(f"Marvis MLX Error (code {result.returncode}): {result.stderr}")
                    raise RuntimeError(f"Marvis synthesis failed: {result.stderr}")
                
                # If streaming, we don't expect a file
                if stream:
                    logger.info("Streaming complete (no file saved)")
                    # Create a placeholder file if output_path is expected
                    # But for now, just return
                    return

                # Find the generated file. It should be output_000.wav
                generated_file = Path(temp_dir) / "output_000.wav"
                
                if generated_file.exists():
                    # Move to final destination
                    shutil.move(str(generated_file), output_path)
                    logger.info(f"Saved Marvis output to {output_path}")
                else:
                    logger.error(f"Expected output file not found: {generated_file}")
                    # List dir to be safe in logs
                    logger.error(f"Temp dir contents: {list(Path(temp_dir).glob('*'))}")
                    raise FileNotFoundError("Marvis output file not found")

        except Exception as e:
            logger.error(f"Marvis synthesis failed: {e}")
            raise e
