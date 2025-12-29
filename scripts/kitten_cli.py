#!/usr/bin/env python3
"""Simple CLI utility to synthesize text using KittenNanoTTS and save to file.

Example:
    python scripts/kitten_cli.py --text "Hello world" --voice expr-voice-4-f --output outputs/kitten.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Kitten TTS Nano CLI - Synthesize text to WAV using KittenNanoTTS")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--model", default="KittenML/kitten-tts-nano-0.2", help="HF model id to use (default: KittenML/kitten-tts-nano-0.2)")
    parser.add_argument("--voice", default="expr-voice-4-f", help="Voice ID to use (default: expr-voice-4-f)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--output", required=True, help="Path to save the generated WAV file")
    parser.add_argument("--cache_dir", default=None, help="Optional cache dir for model files")
    parser.add_argument("--device", default=None, help="Device to run on (cuda/mps/cpu). If not set, autodetects")
    parser.add_argument("--check", action="store_true", help="Run a preflight check to validate dependencies and phonemizer/espeak setup")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Preflight check: run if requested BEFORE attempting to import heavy modules
    if args.check:
        import importlib
        pkgs = ['kittentts', 'phonemizer', 'phonemizer_fork', 'espeakng_loader']
        print('\nPreflight check:')
        for p in pkgs:
            try:
                m = importlib.import_module(p)
                print(f"{p}: version={getattr(m, '__version__', '(no __version__)')}, path={getattr(m, '__file__', '(no file)')}")
            except Exception as exc:
                print(f"{p}: not installed ({exc})")

        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            print('EspeakWrapper: has set_data_path=', hasattr(EspeakWrapper, 'set_data_path'))
            print('EspeakWrapper: has set_library=', hasattr(EspeakWrapper, 'set_library'))
            print('EspeakWrapper: has data_path property=', hasattr(EspeakWrapper, 'data_path'))
            # Show a few attributes to help debugging
            print('Visible attributes:', [a for a in dir(EspeakWrapper) if 'set' in a or 'data' in a or 'library' in a][:30])
        except Exception as exc:
            print('EspeakWrapper: failed to import:', exc)
        sys.exit(0)

    try:
        from voice_cloning.tts.kitten_nano import ensure_espeak_compatibility
        ensure_espeak_compatibility()
        from voice_cloning.tts.kitten_nano import KittenNanoTTS
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise e

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tts = KittenNanoTTS(model_id=args.model, device=args.device, cache_dir=args.cache_dir)
    except Exception as e:
        logger.error(f"Failed to initialize Kitten TTS: {e}")
        sys.exit(1)

    if args.check:
        # Print versions and EspeakWrapper details
        import importlib
        pkgs = ['kittentts', 'phonemizer', 'phonemizer_fork', 'espeakng_loader']
        print('\nPreflight check:')
        for p in pkgs:
            try:
                m = importlib.import_module(p)
                print(f"{p}: version={getattr(m, '__version__', '(no __version__)')}, path={getattr(m, '__file__', '(no file)')}")
            except Exception as exc:
                print(f"{p}: not installed ({exc})")

        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            print('EspeakWrapper: has set_data_path=', hasattr(EspeakWrapper, 'set_data_path'))
            print('EspeakWrapper: has set_library=', hasattr(EspeakWrapper, 'set_library'))
            print('EspeakWrapper: has data_path property=', hasattr(EspeakWrapper, 'data_path'))
        except Exception as exc:
            print('EspeakWrapper: failed to import:', exc)
        sys.exit(0)

    try:
        # Generate audio and write to a file
        tts.synthesize_to_file(args.text, out_path, voice=args.voice, speed=args.speed)
        print(f"✓ Synthesis completed — saved to: {out_path}")
    except Exception as e:
        print(f"✗ Synthesis failed: {e}")
        raise e


if __name__ == "__main__":
    main()
