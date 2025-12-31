import argparse
import sys
import logging
from pathlib import Path

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="Voice Cloning & ASR CLI - Test and compare speech models")
    parser.add_argument("--model", choices=["chatterbox", "kitten", "kitten-0.1", "kitten-0.2", "kokoro", "canary", "parakeet", "granite", "whisper", "humaware", "marvis", "supertone", "neutts-air", "dia2", "cosyvoice", "web"], default="web",
                        help="Model to use (TTS: cosyvoice, chatterbox, kitten[-0.1|-0.2], kokoro, marvis, supertone, neutts-air, dia2 | ASR: canary, parakeet, granite, whisper | VAD: humaware | UI: web). Default: web")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                        help="Device to run model on (cuda/mps/cpu). Auto-detects if not specified.")
    parser.add_argument("--text", type=str, help="Text to synthesize (for TTS models)")
    parser.add_argument("--audio", type=str, help="Audio file to transcribe (for ASR models) or analyze (for VAD models)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--reference", type=str, help="Reference audio file for voice cloning (for models that support it)")
    parser.add_argument("--voice", type=str, help="Voice preset or style")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0)")
    parser.add_argument("--lang-code", type=str, default="e", help="Language code (e.g., 'a' for American English, 'e' for English)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output/playback")
    parser.add_argument("--use-mlx", action="store_true", help="Use MLX backend (Kokoro, Whisper) for Apple Silicon optimization")
    parser.add_argument("--model-id", type=str, help="Specific model ID to use (e.g. openai/whisper-large-v3-turbo)")
    
    # Chatterbox Arguments
    parser.add_argument("--language", default="en", 
                        help="Source language code (Canary/Chatterbox). Default: 'en'")
    parser.add_argument("--target-language", default=None,
                        help="Target language code for translation (Canary only). If not set, performs transcription.")
    # The original speed argument is now replaced by the new one above.
    parser.add_argument("--checkpoints", default="checkpoints_v2", 
                        help="Path to checkpoints directory")
    # The original lang_code and voice arguments are now replaced by the new ones above.
    parser.add_argument("--emotion", default="",
                        help="Emotion tag for Maya1 (e.g., 'laugh', 'sad')")
    parser.add_argument("--exaggeration", type=float, default=0.7,
                        help="Exaggeration factor for Chatterbox (0.0-1.0, default: 0.7)")
    parser.add_argument("--timestamps", action="store_true",
                        help="Enable timestamp output for ASR models (SRT format)")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature for Marvis TTS (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter (default: 0.95)")
    parser.add_argument("--ref_text", 
                        help="Text caption for reference audio (Marvis TTS voice cloning)")
    parser.add_argument("--quantized", action="store_true", default=True,
                        help="Use quantized 4-bit model for faster speed (default: True)")
    parser.add_argument("--no-quantized", action="store_false", dest="quantized",
                        help="Use full precision model")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="CFG guidance weight for Chatterbox (0-1, default: 0.5)")
    parser.add_argument("--multilingual", action="store_true",
                        help="Use Chatterbox multilingual model (23 languages)")
    
    # VAD Arguments
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="Speech detection threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--min-speech-ms", type=int, default=250,
                        help="Minimum speech duration in ms (default: 250)")
    parser.add_argument("--min-silence-ms", type=int, default=100,
                        help="Minimum silence duration in ms (default: 100)")
    parser.add_argument("--speech-pad-ms", type=int, default=30,
                        help="Padding added to speech chunks in ms (default: 30)")
    
    # Supertonic (Supertone) Arguments
    parser.add_argument("--preset", type=str,
                        help="Voice preset for Supertonic TTS")
    parser.add_argument("--steps", type=int, default=8,
                        help="Inference steps for Supertonic (higher = better quality, default: 8)")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="CFG guidance scale for Supertonic (default: 1.0)")
    
    # NeuTTS Air Arguments
    parser.add_argument("--ref-text", type=str,
                        help="Reference text file (transcript of reference audio) for NeuTTS Air")
    parser.add_argument("--backbone", type=str, default="neuphonic/neutts-air-q4-gguf",
                        help="Backbone model for NeuTTS Air (default: neutts-air-q4-gguf)")
    
    # Dia2 Arguments
    




    args = parser.parse_args()

    # Validate text requirement for TTS models
    tts_models = ["chatterbox", "kitten", "kitten-0.1", "kitten-0.2", "kokoro", "marvis", "supertone", "neutts-air", "dia2", "cosyvoice"]
    asr_models = ["canary", "parakeet", "granite", "whisper"]
    vad_models = ["humaware"]
    
    if args.model in tts_models and not args.text:
        print(f"Error: --text is required for TTS models ({', '.join(tts_models)})")
        sys.exit(1)

    # Validate --reference requirement for voice cloning models
    if args.model in ["neutts-air"]:
        if not args.reference:
            print(f"Error: --reference is required for model '{args.model}'")
            sys.exit(1)
        reference_path = Path(args.reference)
        if not reference_path.exists():
            print(f"Error: Reference audio file not found: {args.reference}")
            sys.exit(1)
            
    if args.model in asr_models or args.model in vad_models:
        if not args.reference:
            print(f"Error: --reference is required for {args.model} (path to audio)")
            sys.exit(1)
        ref = Path(args.reference)
        if not ref.exists():
            print(f"Error: Audio file not found: {args.reference}")
            sys.exit(1)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model in tts_models:
        print(f"Text: {args.text}")
    if args.reference:
        label = "Audio" if args.model in asr_models or args.model in vad_models else "Reference"
        print(f"{label}: {args.reference}")
    
    print(f"Output: {args.output}")
    print()

    try:
        if args.model == "web":
            print("Launching Web Interface...")
            from voice_cloning.ui.app import create_interface
            demo = create_interface()
            demo.launch()
            return

        if args.model == "chatterbox":
            print("Loading Chatterbox model...")
            from voice_cloning.tts.chatterbox import synthesize_with_chatterbox
            synthesize_with_chatterbox(
                text=args.text,
                output_wav=args.output,
                source_wav=args.reference,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                language=args.language,
                multilingual=args.multilingual,
                use_mlx=args.use_mlx,
                model_id=args.model_id,
                voice=args.voice
            )
            print(f"✓ Chatterbox synthesis completed! Output saved to: {args.output}")


        elif args.model == "canary":
            print("Loading Canary ASR model...")
            from voice_cloning.asr.canary import transcribe_to_file
            
            # Determine target language (default to source if not specified, which implies transcription)
            target_lang = args.target_language if args.target_language else args.language
            
            transcript_path = transcribe_to_file(
                audio_path=args.reference, 
                output_path=args.output,
                source_lang=args.language,
                target_lang=target_lang
            )
            print(f"✓ Canary transcription completed! Transcript saved to: {transcript_path}")
            
        elif args.model == "parakeet":
            print("Loading Parakeet ASR model...")
            from voice_cloning.asr.parakeet import ParakeetASR
            model = ParakeetASR()
            # Parakeet supports timestamps via MLX backend (SRT format)
            transcript = model.transcribe(args.reference, timestamps=args.timestamps)
            
            # If transcript is an error message
            if transcript.startswith("Error:"):
                print(f"✗ Parakeet transcription failed: {transcript}")
                sys.exit(1)
                
            # Save output
            with open(args.output, "w") as f:
                f.write(transcript)
            print(f"✓ Parakeet transcription completed! Saved to: {args.output}")

        elif args.model == "whisper":
            print("Loading Whisper ASR model...")
            from voice_cloning.asr.whisper import transcribe_to_file
            
            # Determine task (translate if target is en, otherwise transcribe)
            # Whisper only supports X->English translation directly.
            # Determine task (translate if target is en, otherwise transcribe)
            # Whisper only supports X->English translation directly.
            task = "transcribe"
            if args.target_language and args.target_language.lower() == "en" and args.language != "en":
                task = "translate"
                print(f"Task: Translate {args.language} -> English")
            
            # Use provided model ID or default based on MLX flag
            model_id = args.model_id
            if not model_id and args.use_mlx:
                model_id = "mlx-community/whisper-large-v3-turbo"
                print(f"Using default MLX model: {model_id}")
            
            transcribe_to_file(
                audio_path=args.reference,
                output_path=args.output,
                language=args.language if args.language != "en" else None, # Auto-detect if en not strictly enforced? Or just pass it.
                task=task,
                timestamps=args.timestamps,
                use_mlx=args.use_mlx,
                model_id=model_id
            )
            print(f"✓ Whisper transcription completed! Saved to: {args.output}")
            
        elif args.model == "granite":
            print("Loading Granite ASR model...")
            from voice_cloning.asr.granite import transcribe_file
            transcript = transcribe_file(args.reference, args.output)
            print(f"✓ Granite transcription completed! Transcript saved to: {args.output}")
            
        elif args.model == "humaware":
            print("Loading HumAware VAD model...")
            from voice_cloning.vad.humaware import HumAwareVAD
            model = HumAwareVAD()
            segments = model.detect_speech(
                args.reference,
                threshold=args.vad_threshold,
                min_speech_duration_ms=args.min_speech_ms,
                min_silence_duration_ms=args.min_silence_ms,
                speech_pad_ms=args.speech_pad_ms
            )
            # Save segments to text file
            with open(args.output, "w") as f:
                for seg in segments:
                    f.write(f"{seg['start']:.2f} - {seg['end']:.2f}\n")
            print(f"✓ HumAware VAD completed! Segments saved to: {args.output}")
            
        elif args.model == "marvis":
            print("Loading Marvis TTS model...")
            from voice_cloning.tts.marvis import MarvisTTS
            model = MarvisTTS()
            
            # Prepare optional parameters
            kwargs = {}
            if args.reference:
                kwargs['ref_audio'] = args.reference
                print(f"Using voice cloning with reference: {args.reference}")
            if args.ref_text:
                kwargs['ref_text'] = args.ref_text
            if args.stream:
                kwargs['stream'] = True
                print("Streaming mode enabled")
            if args.speed != 1.0:
                kwargs['speed'] = args.speed
            if args.temperature:
                kwargs['temperature'] = args.temperature
            
            # Pass quantized flag
            kwargs['quantized'] = args.quantized
            
            model.synthesize(args.text, args.output, **kwargs)
            print(f"✓ Marvis generation complete! Audio saved to: {args.output}")
        elif args.model in ["kitten", "kitten-0.1", "kitten-0.2"]:
            print("Loading Kitten TTS Nano model...")
            # Ensure phonemizer/espeak compatibility is patched before importing
            try:
                from voice_cloning.tts.kitten_nano import ensure_espeak_compatibility
                ensure_espeak_compatibility()
            except Exception:
                pass
            
            from voice_cloning.tts.kitten_nano import KittenNanoTTS
            
            # Determine version
            model_id = "KittenML/kitten-tts-nano-0.2"  # Default
            if args.model == "kitten-0.1":
                model_id = "KittenML/kitten-tts-nano-0.1"
            elif args.model == "kitten-0.2":
                model_id = "KittenML/kitten-tts-nano-0.2"
                
            print(f"Using model version: {model_id}")
            model = KittenNanoTTS(model_id=model_id)
            
            # Use a valid default voice for Kitten if none provided OR if it's the default 'af_heart' (from Kokoro)
            voice = args.voice
            # If voice is the default for Kokoro ('af_heart'), switch to Kitten default
            if voice == "af_heart":
                voice = "expr-voice-4-f"
                
            model.synthesize_to_file(args.text, args.output, voice=voice, speed=args.speed, stream=args.stream)
            print(f"✓ Kitten TTS synthesis completed! Output saved to: {args.output}")
            
        elif args.model == "supertone":
            print("Loading Supertone (Supertonic) TTS model...")
            from voice_cloning.tts.supertone import synthesize_with_supertone
            
            try:
                result = synthesize_with_supertone(
                    text=args.text,
                    output_path=args.output,
                    preset=args.preset,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    stream=args.stream
                )
                print(f"✓ Supertone synthesis completed! Output saved to: {result}")
            except FileNotFoundError as e:
                print(f"✗ Error: {e}")
                print("\nTo use Supertone, download the models:")
                print("  git lfs install")
                print("  mkdir -p models")
                print("  git clone https://huggingface.co/Supertone/supertonic models/supertonic")
                sys.exit(1)
            except ImportError as e:
                print(f"✗ Error: {e}")
                print("\nInstall onnxruntime:")
                print("  uv pip install onnxruntime")
                sys.exit(1)
        
        elif args.model == "kokoro":
            print("Loading Kokoro TTS model...")
            from voice_cloning.tts.kokoro import synthesize_speech
            
            try:
                # Default voice for Kokoro is af_heart, which is already default in synthesize_speech
                # But if user passed a voice, use it.
                voice = args.voice if args.voice else "af_heart"
                
                result = synthesize_speech(
                    text=args.text,
                    output_path=args.output,
                    voice=voice,
                    speed=args.speed,
                    lang_code=args.lang_code,
                    stream=args.stream,
                    use_mlx=args.use_mlx if hasattr(args, 'use_mlx') else False
                )
                print(f"✓ Kokoro synthesis completed! Output saved to: {result}")
            except Exception as e:
                print(f"✗ Error: {e}")
                sys.exit(1)
        
        elif args.model == "neutts-air":
            print("Loading NeuTTS Air model...")
            from voice_cloning.tts.neutts_air import synthesize_with_neutts_air
            
            # Validate required arguments
            if not args.reference:
                print("✗ Error: --reference is required for NeuTTS Air")
                sys.exit(1)
            
            try:
                result = synthesize_with_neutts_air(
                    text=args.text,
                    output_path=args.output,
                    ref_audio=args.reference,
                    ref_text=args.ref_text,
                    backbone=args.backbone,
                    device="cpu"
                )
                print(f"✓ NeuTTS Air synthesis completed! Output saved to: {result}")
            except FileNotFoundError as e:
                print(f"✗ Error: {e}")
                sys.exit(1)
            except ImportError as e:
                print(f"✗ Error: {e}")
                print("\nMake sure neuttsair module is available.")
                sys.exit(1)
        
        elif args.model == "dia2":
            print("Loading Dia2-1B model...")
            try:
                from voice_cloning.tts.dia2 import Dia2TTS
            except ImportError:
                print("Error: dia2 library not found. Please install it:")
                print("  uv pip install 'dia2 @ git+https://github.com/nari-labs/dia2.git'")
                print("  uv pip install sphn whisper-timestamped")
                sys.exit(1)

            tts = Dia2TTS(device=args.device)
            tts.synthesize(
                text=args.text,
                output_path=args.output,
                cfg_scale=args.cfg_scale if args.cfg_scale else 2.0,
                temperature=args.temperature if args.temperature else 0.8,
                top_k=int(args.top_p) if args.top_p else 50,  # Note: dia2 uses top_k parameter
                verbose=True
            )
            print(f"✓ Dia2 synthesis completed! Output saved to: {args.output}")
            
        elif args.model == "cosyvoice":
            print("Loading CosyVoice2 model...")
            from voice_cloning.tts.cosyvoice import synthesize_speech
            
            # Reference audio is optional generally, but required for MLX backend usually
            # We handle it in the module (fallback), but let's inform user
            ref_audio = args.reference
            if args.use_mlx and not ref_audio:
                 print("Info: MLX CosyVoice2 typically requires reference audio. Module will key default if not provided.")

            result = synthesize_speech(
                text=args.text,
                output_path=args.output,
                ref_audio_path=ref_audio,
                ref_text=args.ref_text, # User can pass reference text via new arg if we added it, or reuse existing args?
                # main.py has args.text, args.reference.
                # marvis uses args.ref_text. Let's assume user passes it.
                # instruct_text isn't in main.py args explicitly as generic 'instruct'.
                # We can map 'emotion' to instruct_text or add a generic 'prompt' arg?
                # For now let's use 'emotion' text if provided as instruct text? Or just leave it for now.
                # Looking at args... args.emotion exists.
                instruct_text=args.emotion if args.emotion else None, 
                speed=args.speed,
                use_mlx=args.use_mlx
            )
            print(f"✓ CosyVoice2 synthesis completed! Output saved to: {result}")


    except ImportError as e:
        print(f"✗ Failed to import {args.model} module: {e}")

    except Exception as e:
        print(f"✗ Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
