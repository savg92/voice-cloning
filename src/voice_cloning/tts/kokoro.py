# """
# This module provides an interface to interact with the Kokoro TTS model.
# Supports both standard PyTorch backend and optimized MLX backend for Apple Silicon.
# """

import soundfile as sf
import os
import subprocess
import tempfile
import logging
from .utils import map_lang_code

logger = logging.getLogger(__name__)


def synthesize_speech(
    text: str,
    audio_sample_path: str | None = None,
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
    Using direct Python API instead of subprocess for better control.
    """
    try:
        import mlx.core as mx  # noqa: F401
        import numpy as np
        from mlx_audio.tts.utils import load_model
    except ImportError:
        raise ImportError(
            "MLX backend requires 'mlx-audio' package. Install with:\n"
            "  pip install mlx-audio\n"
            "Or use use_mlx=False to use PyTorch backend."
        )
    
    # logger.info("Generating speech with MLX backend (Kokoro-82M-4bit)...")
    logger.info("Generating speech with MLX backend (Kokoro-82M-bf16)...")
    logger.info(f"Voice={voice}, Speed={speed}, Lang={lang_code}")

    # Map lang code
    pipeline_lang = map_lang_code(lang_code)

    try:
        # Load model
        # model_path = "mlx-community/Kokoro-82M-4bit"
        model_path = "mlx-community/Kokoro-82M-bf16"
        model = load_model(model_path)
        
        # Generate
        generator = model.generate(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=pipeline_lang
        )
        
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
                        if audio_file is None:
                            break

                        try:
                            # Try afplay (macOS native)
                            subprocess.run(["afplay", audio_file], check=True)
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            try:
                                # Fallback to ffplay
                                subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            except Exception:
                                pass

                        try:
                            os.remove(audio_file)
                        except Exception:
                            pass

                        playback_queue.task_done()
                    except queue.Empty:
                        continue

            player_thread = threading.Thread(target=playback_worker, daemon=True)
            player_thread.start()

            logger.info("Starting streaming generation (MLX)...")
            for result in generator:
                if result.audio is not None:
                    audio_np = np.array(result.audio)
                    all_audio.append(audio_np)

                    # Save chunk and play
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio_np, 24000)
                        playback_queue.put(tmp.name)

            stop_event.set()
            player_thread.join()

        else:
            for result in generator:
                if result.audio is not None:
                    audio_np = np.array(result.audio)
                    all_audio.append(audio_np)
        
        if not all_audio:
             raise RuntimeError("No audio generated by MLX pipeline")

        final_audio = np.concatenate(all_audio)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        sf.write(output_path, final_audio, 24000)
        
        if os.path.exists(output_path):
            logger.info(f"✓ MLX synthesis complete: {output_path}")
            return output_path
        else:
            raise RuntimeError(f"Failed to write MLX output to {output_path}")

    except Exception as e:
        logger.error(f"MLX synthesis failed: {e}")
        raise RuntimeError(f"MLX generation failed: {e}")


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
                    if audio_file is None:
                        break

                    try:
                        subprocess.run(["afplay", audio_file], check=True)
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        try:
                            subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception:
                            pass

                    try:
                        os.remove(audio_file)
                    except Exception:
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
