import unittest
from src.voice_cloning.tts.utils import map_lang_code, KOKORO_LANG_MAP

class TestTTSUtils(unittest.TestCase):
    def test_map_lang_code_common(self):
        self.assertEqual(map_lang_code("en"), "a")
        self.assertEqual(map_lang_code("es"), "e")
        self.assertEqual(map_lang_code("fr"), "f")
        self.assertEqual(map_lang_code("zh"), "z")

    def test_map_lang_code_variants(self):
        self.assertEqual(map_lang_code("en-us"), "a")
        self.assertEqual(map_lang_code("en-gb"), "b")
        self.assertEqual(map_lang_code("pt-br"), "p")

    def test_map_lang_code_case_insensitive(self):
        self.assertEqual(map_lang_code("EN"), "a")
        self.assertEqual(map_lang_code("Es"), "e")

    def test_map_lang_code_unmapped(self):
        # Should return as is if not in map
        self.assertEqual(map_lang_code("ko"), "ko")
        self.assertEqual(map_lang_code("a"), "a")

    def test_map_lang_code_empty(self):
        self.assertEqual(map_lang_code(""), "a")
        self.assertEqual(map_lang_code(None), "a")

if __name__ == "__main__":
    unittest.main()
