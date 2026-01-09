import sys
from voice_cloning.tts.chatterbox_turbo import synthesize_with_chatterbox_turbo
from pathlib import Path
import logging

# Configure logging to see info messages
logging.basicConfig(level=logging.INFO)

def test_multilingual_turbo():
    text = "La captura de Maduro inauguró una nueva forma de diplomacia."
    output_wav = "test_spanish_turbo.wav"
    language = "es"

    print("Testing Chatterbox Turbo Multilingual (MTLTokenizer) for Spanish...")
    try:
        # This should uses the Turbo model with MTLTokenizer, NOT fall back
        synthesize_with_chatterbox_turbo(
            text=text,
            output_wav=output_wav,
            language=language,
            use_mlx=False, # Verify PyTorch backend first
            device="cpu"
        )
        if Path(output_wav).exists():
            print("✓ Synthesis successful! (Check logs to ensure no fallback message appeared)")
        else:
            print("✗ Output file not generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        # Print full stack trace
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_multilingual_turbo()
