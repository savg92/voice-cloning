import logging
import soundfile as sf
import os

logger = logging.getLogger(__name__)

def ensure_espeak_compatibility():
    """
    Helper to patch espeak-ng paths if needed, especially on macOS/Linux
    where phonemizer might not find the library automatically.
    """
    try:
        import os
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        
        # 1. Patch set_data_path if missing (for older misaki/phonemizer versions)
        if not hasattr(EspeakWrapper, 'set_data_path'):
            def set_data_path(path):
                pass
            EspeakWrapper.set_data_path = staticmethod(set_data_path)
            logger.info("Monkey-patched EspeakWrapper.set_data_path")
            
        # 2. Find and set library path
        # Common paths for libespeak-ng on macOS (Homebrew) and Linux
        lib_paths = [
            "/opt/homebrew/lib/libespeak-ng.dylib",
            "/usr/local/lib/libespeak-ng.dylib",
            "/usr/lib/libespeak-ng.so",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"
        ]
        
        for lib in lib_paths:
            if os.path.exists(lib):
                logger.info(f"Using system espeak-ng library at {lib}")
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib
                break
        
        # 3. Find and set data path
        data_paths = [
            "/opt/homebrew/share/espeak-ng-data",
            "/usr/local/share/espeak-ng-data",
            "/usr/share/espeak-ng-data"
        ]
        
        for p in data_paths:
            if os.path.isdir(p):
                logger.info(f"Using system espeak-ng data at {p}")
                os.environ["ESPEAK_DATA_PATH"] = p
                # Also set for newer phonemizer versions
                os.environ["PHONEMIZER_ESPEAK_DATA_PATH"] = p
                break

        # 4. Patch espeakng_loader if present
        try:
            import espeakng_loader
            if not hasattr(espeakng_loader, '_original_get_data_path'):
                espeakng_loader._original_get_data_path = espeakng_loader.get_data_path
                
                def patched_get_data_path():
                    data_path = os.environ.get("ESPEAK_DATA_PATH")
                    if data_path and os.path.isdir(data_path):
                        return data_path
                    return espeakng_loader._original_get_data_path()
                    
                espeakng_loader.get_data_path = patched_get_data_path
                logger.info("Monkey-patched espeakng_loader.get_data_path")
        except ImportError:
            pass
            
    except Exception as e:
        logger.debug(f"Non-critical error in ensure_espeak_compatibility: {e}")

class KittenNanoTTS:
    """
    Wrapper for KittenML/kitten-tts-nano models.
    """
    def __init__(self, model_id: str = "KittenML/kitten-tts-nano-0.2", device: str | None = None, cache_dir: str | None = None):
        self.model_id = model_id
        self.cache_dir = cache_dir
        
        # Ensure compatibility before importing kittentts
        ensure_espeak_compatibility()
        
        try:
            from kittentts import KittenTTS
            from huggingface_hub import hf_hub_download
            
            # Map model_id to specific onnx file
            model_files = {
                "KittenML/kitten-tts-nano-0.1": "kitten_tts_nano_v0_1.onnx",
                "KittenML/kitten-tts-nano-0.2": "kitten_tts_nano_v0_2.onnx"
            }
            onnx_file = model_files.get(model_id, "kitten_tts_nano_v0_2.onnx")
            
            logger.info(f"Downloading/loading Kitten TTS model: {model_id} ({onnx_file})")
            
            model_path = hf_hub_download(repo_id=model_id, filename=onnx_file, cache_dir=cache_dir)
            voices_path = hf_hub_download(repo_id=model_id, filename="voices.npz", cache_dir=cache_dir)
            
            self.model = KittenTTS(model_path=model_path, voices_path=voices_path)
            logger.info(f"Successfully loaded Kitten TTS model from {model_path}")
            
        except ImportError:
            logger.error("Failed to import kittentts or huggingface_hub. Please ensure they are installed.")
            raise
        except Exception as e:
            logger.error(f"Failed to load Kitten TTS model: {e}")
            raise

    def synthesize_to_file(self, text: str, output_path: str, voice: str = "expr-voice-4-f", speed: float = 1.0, stream: bool = False):
        """
        Synthesize text to audio and save to file.
        """
        try:
            # Default voice if None provided
            if not voice:
                voice = "expr-voice-4-f"

            # Default speed if None provided
            if not speed:
                speed = 1.0

            # Ensure text ends with punctuation for better synthesis
            if text and text[-1] not in ".!?":
                text = text + "."

            logger.info(f"Synthesizing with Kitten TTS (voice={voice}, speed={speed})...")
            
            if stream:
                import re
                import threading
                import queue
                import subprocess
                import tempfile
                import numpy as np
                
                # Simple sentence splitter
                sentences = re.split(r'(?<=[.!?])\s+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
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
                
                full_audio_parts = []
                logger.info("Starting streaming generation...")
                
                for sentence in sentences:
                    audio = self.model.generate(sentence, voice=voice, speed=speed)
                    if audio is not None and len(audio) > 0:
                        full_audio_parts.append(audio)
                        
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            sf.write(tmp.name, audio, 24000) # Kitten Nano is 24k? Original file had 24k in comments but 16k in my benchmark assumption.
                            # Let's check the original file again. It had `sf.write('output.wav', audio, 24000)` in comments.
                            # So it is 24k.
                            playback_queue.put(tmp.name)
                
                stop_event.set()
                player_thread.join()
                
                if full_audio_parts:
                    final_audio = np.concatenate(full_audio_parts)
                else:
                    final_audio = np.zeros(24000, dtype=np.float32)
                
                # Pad with silence
                pad_start = int(24000 * 0.2)
                pad_end = int(24000 * 0.5)
                final_audio = np.pad(final_audio, (pad_start, pad_end), 'constant')
                
                sf.write(output_path, final_audio, 24000)
                logger.info(f"Saved audio to {output_path}")
                return

            # Normal non-streaming generation
            audio = self.model.generate(text, voice=voice, speed=speed)

            if audio is None or len(audio) == 0:
                logger.warning("Generated audio is empty! Returning 1 second of silence.")
                import numpy as np
                audio = np.zeros(24000, dtype=np.float32)

            sample_rate = 24000

            # Pad with silence to prevent cutoff (0.2s start, 0.5s end)
            import numpy as np
            pad_start = int(sample_rate * 0.2)
            pad_end = int(sample_rate * 0.5)
            audio = np.pad(audio, (pad_start, pad_end), 'constant')

            logger.info(f"Generated audio shape: {audio.shape}, dtype: {audio.dtype}")

            # Save to file
            sf.write(output_path, audio, sample_rate)
            logger.info(f"Saved audio to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Kitten TTS synthesis failed: {e}")
            raise

