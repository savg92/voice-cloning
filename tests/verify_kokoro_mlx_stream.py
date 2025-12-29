import os
import logging
from src.voice_cloning.tts.kokoro import synthesize_speech

logging.basicConfig(level=logging.INFO)

def test_kokoro_mlx_stream():
    text = "This is a streaming test for Kokoro MLX. It should play audio as it generates."
    output_path = "outputs/test_kokoro_stream_mlx.wav"
    
    os.makedirs("outputs", exist_ok=True)
    
    print("Testing Kokoro MLX with Streaming...")
    try:
        result = synthesize_speech(
            text=text,
            output_path=output_path,
            lang_code="a",
            voice="af_heart",
            use_mlx=True,
            stream=True
        )
        print(f"✓ Success! (Result path: {result})")
    except Exception as e:
        print(f"✗ Failed: {e}")

if __name__ == "__main__":
    test_kokoro_mlx_stream()
