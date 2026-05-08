import os
import unittest
from pathlib import Path
import torch
from src.voice_cloning.tts.omnivoice import OmniVoiceTTS

class TestOmniVoiceTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path("test_results/omnivoice")
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            cls.tts = OmniVoiceTTS()
        except Exception as e:
            raise unittest.SkipTest(f"Failed to load OmniVoice model: {e}")

    def test_basic_synthesis(self):
        output_path = self.output_dir / "test_basic.wav"
        result = self.tts.synthesize(
            text="Hello from OmniVoice unit test.",
            output_path=str(output_path)
        )
        self.assertTrue(os.path.exists(result))
        self.assertEqual(result, str(output_path))

    def test_voice_design(self):
        output_path = self.output_dir / "test_design.wav"
        result = self.tts.synthesize(
            text="Testing voice design feature.",
            instruct="male, young adult, low pitch",
            output_path=str(output_path)
        )
        self.assertTrue(os.path.exists(result))

    def test_multilingual(self):
        # Testing French synthesis
        output_path = self.output_dir / "test_french.wav"
        result = self.tts.synthesize(
            text="Bonjour tout le monde.",
            language="fr",
            output_path=str(output_path)
        )
        self.assertTrue(os.path.exists(result))

if __name__ == "__main__":
    unittest.main()
