# """
# This module provides an interface to interact with the Kokoro TTS model.
# Supports both standard PyTorch backend and optimized MLX backend for Apple Silicon.
# """

import soundfile as sf
import os
import shutil
import subprocess
import tempfile
import sys
import logging
from typing import Optional
from .utils import map_lang_code

logger = logging.getLogger(__name__)


def synthesize_speech(
    text: str,
    audio_sample_path: Optional[str] = None,
    output_path: str = "kokoro_output.wav",
    lang_code: str = "a",
    voice: str = "af_heart",
    speed: float = 1.0,
    stream: bool = False,
    use_mlx: bool = False
) -> str:
    """
    Synthesizes speech using the Kokoro model.

    Args:
        text (str): The text to be converted to speech.
        audio_sample_path (Optional[str]): Path to the audio sample for voice cloning (not used in basic Kokoro pipeline).
        output_path (str): Path to save the generated audio file.
        lang_code (str): Language code for the TTS pipeline (default: 'a' for US English).
        voice (str): Voice name to use (default: 'af_heart').
        speed (float): Speech speed (default: 1.0).
        stream (bool): Enable streaming playback.
        use_mlx (bool): Use MLX backend for Apple Silicon optimization (default: False).

    Returns:
        str: Path to the generated audio file.
    """

    if use_mlx:
        return _synthesize_with_mlx(text, output_path, lang_code, voice, speed, stream)
    else:
        return _synthesize_with_pytorch(text, output_path, lang_code, voice, speed, stream)


def _synthesize_with_mlx(
    text: str,
    output_path: str,
    lang_code: str,
    voice: str,
    speed: float,
    stream: bool
) -> str:
    """
    Synthesize speech using MLX backend (optimized for Apple Silicon).
    """
    try:
        import mlx_audio
    except ImportError:
        raise ImportError(
            "MLX backend requires 'mlx-audio' package. Install with:\n"
            "  pip install mlx-audio\n"
            "Or use use_mlx=False to use PyTorch backend."
        )
    
    logger.info("Generating speech with MLX backend (Kokoro-82M-bf16)...")
    logger.info(f"Voice={voice}, Speed={speed}, Lang={lang_code}")
    
    # MLX audio uses command-line interface
    with tempfile.TemporaryDirectory() as tmpdir:
        # mlx-audio generates output with file_prefix
        file_prefix = os.path.join(tmpdir, "mlx_output")
        
        # Apply language mapping if needed (consistent with PyTorch backend)
        mapped_lang = map_lang_code(lang_code)

        cmd = [
            sys.executable, "-m", "mlx_audio.tts.generate",
            "--model", "mlx-community/Kokoro-82M-bf16",
            "--text", text,
            "--voice", voice,
            "--speed", str(speed),
            "--lang_code", mapped_lang,
            "--file_prefix", file_prefix
        ]
        
        if stream:
            cmd.append("--stream")
            # If we are streaming, we might not get a file, or it might be partial.
            # mlx-audio generate --stream usually plays audio.
        
        logger.info(f"Running MLX TTS command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if stream:
                # For streaming, we return a success indicator or similar
                logger.info("MLX Streaming finished.")
                return output_path

            # MLX generates file_prefix_000.wav (with sequence number)
            generated_file = f"{file_prefix}_000.wav"
            if not os.path.exists(generated_file):
                # Try without sequence for older versions
                generated_file = f"{file_prefix}.wav"
                if not os.path.exists(generated_file):
                    # Check if maybe it's 24k or something else
                    logger.error(f"MLX stdout: {result.stdout}")
                    logger.error(f"MLX stderr: {result.stderr}")
                    raise RuntimeError(f"MLX did not generate expected output file")
            
            # Move to final output
            shutil.move(generated_file, output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MLX synthesis failed: {e.stderr}")
            raise RuntimeError(f"MLX generation failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("mlx_audio.tts.generate not found. Ensure mlx-audio is properly installed.")
    
    logger.info(f"✓ MLX synthesis complete: {output_path}")
    return output_path


def _synthesize_with_pytorch(
    text: str,
    output_path: str,
    lang_code: str,
    voice: str,
    speed: float,
    stream: bool
) -> str:
    """
    Synthesize speech using PyTorch backend (standard).
    """
    from kokoro import KPipeline
    import torch
    import numpy as np

    # Map common language codes to Kokoro codes if needed
    pipeline_lang = map_lang_code(lang_code)

    try:
        pipeline = KPipeline(lang_code=pipeline_lang, repo_id="hexgrad/Kokoro-82M")
    except Exception as e:
        logger.error(f"Error initializing Kokoro pipeline with lang='{pipeline_lang}': {e}")
        raise

    generator = pipeline(text, voice=voice, speed=speed)

    all_audio = []

    if stream:
        import threading
        import queue

        playback_queue = queue.Queue()
        stop_event = threading.Event()

        def playback_worker():
            while not stop_event.is_set() or not playback_queue.empty():
                try:
                    audio_file = playback_queue.get(timeout=0.1)
                    if audio_file is None: break

                    try:
                        subprocess.run(["afplay", audio_file], check=True)
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        try:
                            subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except:
                            pass

                    try:
                        os.remove(audio_file)
                    except:
                        pass

                    playback_queue.task_done()
                except queue.Empty:
                    continue

        player_thread = threading.Thread(target=playback_worker, daemon=True)
        player_thread.start()

        logger.info("Starting streaming generation...")
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                all_audio.append(audio)

                # Save chunk and play
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio, 24000)
                    playback_queue.put(tmp.name)

        stop_event.set()
        player_thread.join()

    else:
        for i, (gs, ps, audio) in enumerate(generator):
            if audio is not None:
                all_audio.append(audio)

    if not all_audio:
        logger.warning("No audio generated")
        return None

    final_audio = np.concatenate(all_audio)
    logger.info(f"Writing final audio ({len(final_audio)} samples) to {output_path}")

    sf.write(output_path, final_audio, 24000)
    
    if os.path.exists(output_path):
        logger.info(f"✓ Successfully wrote {output_path}")
    else:
        logger.error(f"✗ Failed to write {output_path}")

    return output_path
