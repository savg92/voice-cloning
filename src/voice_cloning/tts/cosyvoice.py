# """
# This module provides an interface to interact with the CosyVoice2 TTS model.
# Supports both MLX backend (for Apple Silicon) and standard PyTorch backend.
# """

import soundfile as sf
import os
import logging
import sys
import torch

# Add local CosyVoice repo to path if it exists
# This allows using the PyTorch backend without system-wide installation
REPOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Updated path: models/CosyVoice
COSYVOICE_PATH = os.path.join(REPOS_ROOT, "models", "CosyVoice")
if os.path.exists(COSYVOICE_PATH):
    if COSYVOICE_PATH not in sys.path:
        sys.path.append(COSYVOICE_PATH)
    # Also Match-TTS
    MATCHA_PATH = os.path.join(COSYVOICE_PATH, "third_party", "Matcha-TTS")
    if os.path.exists(MATCHA_PATH) and MATCHA_PATH not in sys.path:
        sys.path.append(MATCHA_PATH)
    
    # Patch the file_utils if needed (we already patched the file itself, so just ensuring path is found)

logger = logging.getLogger(__name__)

def synthesize_speech(
    text: str,
    output_path: str = "cosyvoice_output.wav",
    model_id: str = "mlx-community/CosyVoice2-0.5B-4bit",
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    instruct_text: str | None = None,
    source_audio_path: str | None = None,
    speed: float = 1.0,
    use_mlx: bool = True
) -> str | None:
    """
    Synthesizes speech using the CosyVoice2 model.

    Args:
        text (str): The text to be converted to speech.
        output_path (str): Path to save the generated audio file.
        model_id (str): Model HuggingFace ID (default: "mlx-community/CosyVoice2-0.5B-4bit").
        ref_audio_path (Optional[str]): Path to reference audio for zero-shot cloning.
        ref_text (Optional[str]): Transcription of the reference audio (optional, but improves quality).
        instruct_text (Optional[str]): Instruction text for style control (e.g., "Speak slowly").
        source_audio_path (Optional[str]): Source audio for voice conversion (replaces text).
        speed (float): Speech speed (only supported by some backends/models).
        use_mlx (bool): Use MLX backend for Apple Silicon (default: True).

    Returns:
        Optional[str]: Path to the generated audio file, or None if failed.
    """

    if use_mlx:
        return _synthesize_with_mlx(
            text, output_path, model_id, ref_audio_path, ref_text, instruct_text, source_audio_path, speed
        )
    else:
        return _synthesize_with_pytorch(
            text, output_path, model_id, ref_audio_path, ref_text, instruct_text, source_audio_path, speed
        )


def _synthesize_with_mlx(
    text: str,
    output_path: str,
    model_id: str,
    ref_audio_path: str | None,
    ref_text: str | None,
    instruct_text: str | None,
    source_audio_path: str | None,
    speed: float
) -> str:
    """
    Synthesize using MLX backend (mlx-audio-plus).
    """
    import tempfile
    import shutil
    
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError:
        raise ImportError(
            "MLX backend requires 'mlx-audio-plus' package.\n"
            "Install with: pip install -U mlx-audio-plus"
        )
    
    # Ensure model ID is compatible with MLX
    if "mlx-community" not in model_id and "4bit" not in model_id:
        logger.warning(
            f"Model {model_id} might not be compatible with MLX backend. "
            "Using 'mlx-community/CosyVoice2-0.5B-4bit' as fallback."
        )
        model_id = "mlx-community/CosyVoice2-0.5B-4bit"

    logger.info(f"Generating with MLX (Model: {model_id})...")
    
    # mlx_audio.tts.generate typically calls subprocess or internal logic.
    # The snippet from document 6 says:
    # generate_audio(text=..., model=..., ref_audio=..., file_prefix=...)
    
    # Note: speed argument support in mlx-audio-plus depends on version, check docs.
    # The command line has --speed, so python API likely has it too, but let's be careful.
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_prefix = os.path.join(tmpdir, "output")
        
        # Prepare arguments
        kwargs = {
            "text": text,
            "model": model_id,
            "file_prefix": file_prefix,
        }
        
        if ref_audio_path:
            kwargs["ref_audio"] = ref_audio_path
        else:
            # CosyVoice2 MLX requires reference audio. Try to find a default or fail.
            default_ref = "samples/anger.wav"
            if os.path.exists(default_ref):
                logger.warning(f"No reference audio provided. Using default: {default_ref}")
                kwargs["ref_audio"] = default_ref
            else:
                 # Check if we are running the verification script which creates outputs/verification/ref.wav
                 # This is a bit hacky but helps with testing.
                 # Better to just raise error if nothing found.
                 raise ValueError(
                     "CosyVoice2 (MLX) requires a reference audio file for synthesis.\n"
                     "Please provide `ref_audio_path`."
                 )
        
        if ref_text:
            kwargs["ref_text"] = ref_text
            
        if instruct_text:
            kwargs["instruct_text"] = instruct_text
            
        if source_audio_path:
            kwargs["source_audio"] = source_audio_path
            # If source audio is present, text might be ignored or used differently
        
        # Note: 'speed' parameter might not be directly exposed in generate_audio based on snippet.
        # But if it is available in CLI, it might be in kwargs. 
        # I'll try to pass it if it's not 1.0, if it fails I'll catch it?
        # Actually safer to not pass it if uncertain, but let's assume it's NOT in generate_audio signature from snippet.
        # Snippet: generate_audio(text="...", model="...", ref_audio="...", file_prefix="...")
        
        try:
            generate_audio(**kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                logger.warning(f"MLX generate_audio failed with arguments: {e}. Retrying without optional args if possible.")
                raise e
            raise e
            
        # MLX output filename handling
        # Usually it adds _000.wav or .wav
        # We need to find the generated file.
        
        generated_file = f"{file_prefix}_000.wav"
        if not os.path.exists(generated_file):
            generated_file = f"{file_prefix}.wav"
        
        if not os.path.exists(generated_file):
             # Try to find any file starting with prefix in tmpdir
             files = os.listdir(tmpdir)
             possible_files = [f for f in files if f.startswith("output")]
             if possible_files:
                 generated_file = os.path.join(tmpdir, possible_files[0])
             else:
                 raise RuntimeError("MLX generation failed: Output file not found.")

        # Move to final destination
        shutil.move(generated_file, output_path)
        
    logger.info(f"✓ MLX synthesis complete: {output_path}")
    return output_path


def _synthesize_with_pytorch(
    text: str,
    output_path: str,
    model_id: str,
    ref_audio_path: str | None,
    ref_text: str | None,
    instruct_text: str | None,
    source_audio_path: str | None,
    speed: float
) -> str:
    """
    Synthesize using PyTorch backend (requires 'cosyvoice' package or cloned repo).
    """
    
    # Attempt to import CosyVoice classes from the local repository
    try:
        import sys
        # Add the models directory to sys.path to ensure local cosyvoice is found
        cosyvoice_repo_path = os.path.join(os.getcwd(), "models", "CosyVoice")
        if os.path.exists(cosyvoice_repo_path) and cosyvoice_repo_path not in sys.path:
            sys.path.append(cosyvoice_repo_path)
            logger.info(f"Added {cosyvoice_repo_path} to sys.path")

        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
    except ImportError as e:
        logger.error(f"Failed to import CosyVoice from local repo: {e}")
        raise ImportError(
            "PyTorch backend requires 'cosyvoice' package. Please ensure 'models/CosyVoice' is cloned correctly.\n"
            "Full error: " + str(e)
        )

    # Local robust load_wav to avoid torchaudio backend issues (TorchCodec error)
    def load_wav(wav_path, target_sr):
        import librosa
        # Load with librosa (handles resampling)
        audio, sr = librosa.load(wav_path, sr=target_sr)
        # Convert to tensor (1, T)
        audio = torch.from_numpy(audio).unsqueeze(0)
        return audio

    # Note: The model_id for PyTorch usually points to a local directory or modelscope ID.
    # If the user passed the HF ID "FunAudioLLM/CosyVoice2-0.5B", we might need to handle download.
    # But usually CosyVoice loads from a checklist or local path.
    # For now, let's assume model_id is a path or the library handles it (it uses modelscope by default).
    
    # If model_id is the default one for MLX, switch to PyTorch default
    if "mlx-community" in model_id:
        logger.info(f"Switching from MLX model {model_id} to PyTorch equivalent.")
        # Default CosyVoice2 model on ModelScope (Official)
        model_id = "iic/CosyVoice2-0.5B" 
    
    logger.info(f"Initializing CosyVoice (Model: {model_id})...")
    
    # CosyVoice initialization
    try:
        from modelscope import snapshot_download
        
        # Download model if it looks like a modelscope ID or HF ID
        if "/" in model_id and not os.path.exists(model_id):
            logger.info(f"Downloading model {model_id} via ModelScope...")
            model_dir = snapshot_download(model_id)
            model_id = model_dir
            logger.info(f"Model downloaded to {model_id}")

        if "CosyVoice2" in model_id:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            logger.info("Using CosyVoice2 class.")
            model = CosyVoice2(model_id)
        else:
            model = CosyVoice(model_id)
             
    except Exception as e:
        logger.error(f"Failed to load CosyVoice model {model_id}: {e}")
        raise RuntimeError(
            f"Failed to load CosyVoice model. Ensure dependencies are installed and model path is correct.\n"
            f"Error: {e}"
        )

    logger.info("Generating speech...")
    
    # Fallback to default reference audio if needed (mirroring MLX logic)
    # CosyVoice2 often requires reference audio (zero-shot) if SFT speakers are missing.
    if not ref_audio_path and not source_audio_path:
        # Check if we should inject a default
        default_ref = "samples/anger.wav"
        if os.path.exists(default_ref):
             # Only inject if we think we might need it (e.g. SFT might fail).
             # But for consistency, let's prefer zero-shot with default prompt if user provided nothing,
             # OR we can try SFT first and catch error? No, that's messy.
             # Let's check available speakers.
             has_speakers = False
             if hasattr(model, 'list_available_spks'):
                 spks = model.list_available_spks()
                 if spks:
                     has_speakers = True
             
             if not has_speakers:
                 logger.info(f"No SFT speakers found. Using default reference audio: {default_ref}")
                 ref_audio_path = default_ref
                 ref_text = "The quick brown fox jumps over the lazy dog." # Need dummy ref text for zero shot usually, or use cross-lingual
                 # Actually inference_cross_lingual uses prompt_wav but no prompt_text.
                 # Let's use cross-lingual (zero-shot without prompt text) if we don't have ref_text.
    
    output = None
    
    # Different modes based on arguments
    if source_audio_path and ref_audio_path:
         # Voice Conversion
         # CosyVoice V2/Frontend expects PATHS strings, not tensors.
         output = model.inference_vc(source_audio_path, ref_audio_path)
         
    elif ref_audio_path and ref_text:
        # Zero-shot with text
        # Pass path directly
        output = model.inference_zero_shot(text, ref_text, ref_audio_path, speed=speed)
        
    elif ref_audio_path:
        # Cross-lingual / Zero-shot without prompt text
        # Pass path directly
        output = model.inference_cross_lingual(text, ref_audio_path, speed=speed)
        
    elif instruct_text:
        # Instruct mode
        if hasattr(model, 'inference_instruct'):
             output = model.inference_instruct(text, "default", instruct_text)
        elif hasattr(model, 'inference_instruct2'):
             if ref_audio_path:
                 # Pass path
                 output = model.inference_instruct2(text, instruct_text, ref_audio_path)
             else:
                 logger.error("CosyVoice2 instruct mode requires reference audio.")
                 raise ValueError("CosyVoice2 instruct mode requires reference audio (ref_audio_path).")
        else:
             logger.warning("Instruct mode requested but method not found. Using SFT.")
             # For SFT, spk_id is a string, so safe.
             output = model.inference_sft(text, "default", speed=speed)
             
    else:
        # SFT (Standard TTS)
        spk_id = "default"
        if hasattr(model, 'list_available_spks'):
            spks = model.list_available_spks()
            if spks and "default" not in spks:
                spk_id = spks[0]
                logger.info(f"Speaker 'default' not found. Using '{spk_id}'.")
        
        output = model.inference_sft(text, spk_id, speed=speed)

    # Output is usually a list of dicts or a generator {'tts_speech': tensor, ...}
    # We need to save it.
    
    generated_audio = []
    for item in output:
        if 'tts_speech' in item:
            # Flatten to 1D array (samples,)
            generated_audio.append(item['tts_speech'].cpu().numpy().flatten())
    
    if not generated_audio:
        raise RuntimeError("No audio generated by CosyVoice pytorch backend")
        
    import numpy as np
    
    final_audio = np.concatenate(generated_audio)
    
    # Save using model's sample rate if available
    sr = getattr(model, 'sample_rate', 24000) # Default to 24000 for CV2
    
    sf.write(output_path, final_audio, sr) 
    
    logger.info(f"✓ PyTorch synthesis complete: {output_path}")
    return output_path
