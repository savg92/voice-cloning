import os
from unittest.mock import patch, MagicMock

# Mock spacy and subprocess BEFORE importing anything else
import spacy
spacy.util.is_package = MagicMock(return_value=True)
spacy.load = MagicMock(return_value=spacy.blank("en"))
spacy.cli = MagicMock()
spacy.cli.download = MagicMock()

def mocked_check_call(*args, **kwargs):
    print(f"Mocking check_call for: {args}")
    return 0

with patch("subprocess.check_call", side_effect=mocked_check_call):
    from mlx_audio.tts.generate import generate_audio
    import logging

    logging.getLogger("mlx_audio").setLevel(logging.ERROR)

    def generate_voice_samples():
        output_dir = "/Users/savg/Desktop/voice-cloning/samples/kokoro_voices"
        os.makedirs(output_dir, exist_ok=True)
        
        # Target: Generating voices for the remaining Turbo supported languages
        # Languages already done: en, es, fr, it, pt
        # Remaining: de, zh, ja, hi, ko, tr, ru, vi, he, etc.
        # Note: ja/zh might still fail due to missing native libs, but we'll try what we can.
        
        voices = [
            ("df_sarah", "de", "Der schnelle braune Fuchs springt über den faulen Hund. Dies ist eine konsistente deutsche Stimme."),
            ("ru_0", "ru", "Быстрая коричневая лиса прыгает через ленивую собаку. Это образец русского голоса."),
            ("hf_alpha", "hi", "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद गई। यह एक सुसंगत आवाज नमूना है।"),
            ("tr_0", "tr", "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar. Bu tutarlı bir ses örneğidir."),
            ("ko_0", "ko", "빠른 갈색 여우가 게으른 개를 뛰어넘습니다. 이것은 일관된 음성 샘플입니다."),
            ("vi_0", "vi", "Con cáo nâu nhanh nhẹn nhảy qua con chó lười biếng. Đây là một mẫu giọng nói nhất quán."),
            # Arabic, Hebrew etc might not have built-in Kokoro voices in all models, but let's check
            # af_heart is generic, but some models have language specific ones
        ]

        for voice, lang, text in voices:
            # We check if we already have it to save time
            target_path = os.path.join(output_dir, f"{voice}.wav")
            if os.path.exists(target_path):
                print(f"Skipping {voice}, already exists.")
                continue

            print(f"Generating Kokoro sample for {voice} (lang: {lang})...")
            try:
                file_prefix = os.path.join(output_dir, voice)
                generate_audio(
                    text=text,
                    voice=voice,
                    lang_code=lang,
                    file_prefix=file_prefix,
                    verbose=False
                )
                
                gen_path = f"{file_prefix}_000.wav"
                if os.path.exists(gen_path):
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.rename(gen_path, target_path)
                    print(f"✓ Saved {target_path}")
                else:
                     print(f"✗ Failed to find generated file: {gen_path}")
            except Exception as e:
                print(f"✗ Failed to generate {voice}: {e}")

    if __name__ == "__main__":
        generate_voice_samples()
