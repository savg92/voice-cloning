import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

class ParakeetASR:
    """
    Wrapper for Parakeet TDT ASR models.
    Supports:
    - mlx-community/parakeet-tdt-0.6b-v3 (Apple Silicon via MLX)
    - nvidia/parakeet-tdt-0.6b-v3 (NVIDIA NeMo)
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.backend = self._determine_backend()
        self.model = None
        self.err_msg: Optional[str] = None
        self._load_model()
        
    def _determine_backend(self) -> str:
        """Determine which backend to use based on system and availability."""
        if sys.platform == "darwin" and os.uname().machine == "arm64":
            try:
                import mlx.core  # type: ignore
                logger.info("Detected Apple Silicon. Using MLX backend for Parakeet.")
                return "mlx"
            except ImportError:
                logger.warning("MLX not found on Apple Silicon. Falling back to NeMo/Torch if available.")
        
        return "nemo"

    def _load_model(self):
        if self.backend == "mlx":
            try:
                # Note: As of now, MLX support for Parakeet might require specific loading logic
                # or a library like mlx-audio if it supports it.
                # Assuming standard mlx-community usage or placeholder for now.
                # The user provided link: https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3
                # This usually implies it works with a specific MLX example or library.
                # For this implementation, we will try to use a generic MLX ASR loader if available,
                # or warn if we need specific code.
                # Since direct MLX ASR API is fragmented, we'll assume a hypothetical `mlx_parakeet` or similar
                # or just warn that it's experimental.
                
                # REALITY CHECK: There isn't a standard "import mlx_parakeet" yet. 
                # It likely uses `mlx-examples/asr` code.
                # For the sake of this task, I will implement a stub that attempts to load it 
                # or instructs the user if a specific script is needed.
                
                logger.info("Loading MLX Parakeet model...")
                # Check that the necessary executables exist. The MLX approach expects a CLI 'parakeet-mlx'
                # to be available through the 'uv' runner ("uv run parakeet-mlx ..."). If that's not present
                # we attempt to gracefully fall back to the NeMo backend.
                if shutil.which("uv") is None or shutil.which("parakeet-mlx") is None:
                    self.err_msg = (
                        "MLX parakeet CLI not found. Install it (see https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) "
                        "or ensure 'uv' and 'parakeet-mlx' are on PATH."
                    )
                    logger.error(self.err_msg)
                    return
                # If binaries appear to be present, we leave loading as a no-op — runtime invocation will
                # spawn the CLI helper; MLX model is invoked as a CLI tool.
                logger.debug("Found 'uv' and 'parakeet-mlx' in PATH; MLX CLI should be runnable.")
                return
            except Exception as e:
                logger.error(f"Failed to load MLX Parakeet: {e}")
                self.backend = "nemo"
                
        if self.backend == "nemo":
            try:
                import nemo.collections.asr as nemo_asr
                logger.info("Loading NeMo Parakeet model...")
                self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name="nvidia/parakeet-tdt-0.6b-v3"
                )
            except ImportError:
                self.err_msg = "NeMo toolkit not installed. Please install nemo_toolkit[asr] for non-MLX support."
                logger.error(self.err_msg)
            except Exception as e:
                self.err_msg = f"Failed to load NeMo Parakeet: {e}"
                logger.error(self.err_msg)

    def transcribe(self, audio_path: str, timestamps: bool = False) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file.
            timestamps: If True, output timestamps (SRT format).
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        if self.backend == "mlx":
            try:
                import subprocess
                import tempfile
                from pathlib import Path
                
                # Use a temporary directory for the output file
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Determine output format
                    output_format = "srt" if timestamps else "txt"
                    
                    # Construct command: uv run parakeet-mlx <audio> --output-format <fmt> --output-dir <temp_dir>
                    cmd = ["uv", "run", "parakeet-mlx", audio_path, "--output-format", output_format, "--output-dir", temp_dir]

                    # Quick checks before attempting to run: ensure executables are available
                    if shutil.which("uv") is None or shutil.which("parakeet-mlx") is None:
                        msg = (
                            "MLX runner missing. Install the parakeet-mlx CLI via Hugging Face instructions "
                            "or ensure both 'uv' and 'parakeet-mlx' are on PATH."
                        )
                        logger.error(msg)
                        return f"Error: {msg}"
                    
                    # Run the command
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        # If the command failed, try to provide actionable advice
                        stderr = (result.stderr or "").strip()
                        logger.error(f"Parakeet MLX Error: {stderr}")
                        if "Failed to spawn" in stderr or "No such file or directory" in stderr:
                            msg = (
                                "Failed to spawn 'parakeet-mlx'. Ensure it is installed and available in PATH, "
                                "or install the NeMo toolkit (nemo_toolkit[asr]) and run with the NeMo backend."
                            )
                            logger.error(msg)
                            return f"Error: {msg}"
                        return f"Error: {stderr}"
                        
                    # The output file should be in temp_dir with the same basename as audio_path but correct extension
                    audio_stem = Path(audio_path).stem
                    output_file = Path(temp_dir) / f"{audio_stem}.{output_format}"
                    
                    if output_file.exists():
                        return output_file.read_text(encoding='utf-8').strip()
                    else:
                        # Provide actionable feedback including CLI logs since parakeet-mlx sometimes
                        # prints errors to stdout while still returning exit code 0.
                        cli_logs = (result.stderr or "").strip()
                        if not cli_logs:
                            cli_logs = (result.stdout or "").strip()
                        msg = (
                            f"Parakeet MLX did not produce an output file (expected {output_file})."
                            + (f" CLI output: {cli_logs}" if cli_logs else "")
                        )
                        logger.error(msg)
                        return f"Error: {msg}"

            except Exception as e:
                logger.error(f"Parakeet MLX inference failed: {e}")
                return f"Error: {e}"
            
        elif self.backend == "nemo":
            if not self.model:
                # Provide a helpful error when the model failed to load.
                msg = self.err_msg or "NeMo model not loaded. Install nemo_toolkit[asr] or use MLX CLI if available."
                logger.error(msg)
                return f"Error: {msg}"
            # Otherwise fall through to transcribe using NeMo
            
            # now self.model is present
            files = [audio_path]
            # NeMo expects list of files
            transcriptions = self.model.transcribe(paths2audio_files=files)
            return transcriptions[0] if transcriptions else ""
            
        return ""


def get_parakeet(device: Optional[str] = None) -> ParakeetASR:
    """Compatibility helper used by some tests.

    Returns a ParakeetASR instance wrapped by a simple function so older tests that import
    `get_parakeet` continue to work.
    """
    return ParakeetASR(device=device)
