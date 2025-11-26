# """
# This module provides an interface to interact with the Kokoro TTS model.
# """

import soundfile as sf
from kokoro import KPipeline
import torch
from typing import Optional

def synthesize_speech(text: str, audio_sample_path: Optional[str] = None, output_path: str = "kokoro_output.wav", lang_code: str = "e", voice: str = "af_heart", speed: float = 1.0, stream: bool = False) -> str:
    """
    Synthesizes speech using the Kokoro model.

    Args:
        text (str): The text to be converted to speech.
        audio_sample_path (Optional[str]): Path to the audio sample for voice cloning (not used in basic Kokoro pipeline).
        output_path (str): Path to save the generated audio file.
        lang_code (str): Language code for the TTS pipeline (default: 'a' for American English).
        voice (str): Voice name to use (default: 'af_heart').
        speed (float): Speech speed (default: 1.0).
        stream (bool): Enable streaming playback.

    Returns:
        str: Path to the generated audio file.
    """
    use_cuda = torch.cuda.is_available()
    
    # Map common language codes to Kokoro codes if needed
    lang_map = {
        'en-us': 'a', 'en': 'a',
        'en-gb': 'b', 'en-uk': 'b',
        'fr': 'f', 'fr-fr': 'f',
        'ja': 'j', 'jp': 'j',
        'zh': 'z', 'cn': 'z',
        'es': 'e',
        'it': 'i',
        'pt': 'p', 'pt-br': 'p',
        'hi': 'h'
    }
    
    pipeline_lang = lang_map.get(lang_code.lower(), lang_code)
    
    try:
        pipeline = KPipeline(lang_code=pipeline_lang, repo_id="hexgrad/Kokoro-82M")
    except Exception as e:
        print(f"Error initializing Kokoro pipeline with lang='{pipeline_lang}': {e}")
        raise

    generator = pipeline(text, voice=voice, speed=speed)
    
    import numpy as np
    all_audio = []
    
    if stream:
        import threading
        import queue
        import subprocess
        import tempfile
        import os
        
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
        
        print("Starting streaming generation...")
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
        print("Warning: No audio generated.")
        return None
        
    final_audio = np.concatenate(all_audio)
    
    sf.write(output_path, final_audio, 24000)
    return output_path

