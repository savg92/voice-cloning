import os
import logging
from voice_cloning.tts.kokoro import synthesize_speech

logging.basicConfig(level=logging.INFO)

def test_kokoro_mlx_spanish():
    text = "Hola, esto es una prueba del sistema de clonación de voz en español."
    output_path = "outputs/test_kokoro_es_mlx.wav"
    
    # Ensure output dir exists
    os.makedirs("outputs", exist_ok=True)
    
    print("Testing Kokoro MLX with Spanish...")
    try:
        # 'e' is the code for Spanish in Kokoro
        result = synthesize_speech(
            text=text,
            output_path=output_path,
            lang_code="e",
            voice="ef_dora", # Standard Spanish voice
            use_mlx=True
        )
        print(f"✓ Success! Audio saved to: {result}")
    except Exception as e:
        print(f"✗ Failed: {e}")

if __name__ == "__main__":
    test_kokoro_mlx_spanish()
