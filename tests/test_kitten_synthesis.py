import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.getcwd())

from src.voice_cloning.tts.kitten_nano import KittenNanoTTS

def test_synthesis():
    try:
        print("Initializing KittenNanoTTS...")
        model = KittenNanoTTS()
        
        output_file = "outputs/kitten_test.wav"
        text = "This is a test of the Kitten TTS Nano model."
        
        print(f"Synthesizing text: '{text}'")
        model.synthesize_to_file(text, output_file, voice="expr-voice-4-f", speed=1.0)
        
        if os.path.exists(output_file):
            print(f"✓ Successfully generated {output_file}")
            # Optional: Check file size to ensure it's not empty
            size = os.path.getsize(output_file)
            print(f"  File size: {size} bytes")
        else:
            print(f"✗ Failed to generate {output_file}")
            
    except Exception as e:
        print(f"✗ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_synthesis()
