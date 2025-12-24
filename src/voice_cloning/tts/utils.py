# Map common language codes to Kokoro internal codes
# Used by Kokoro, Marvis, and Chatterbox (MLX backends)
KOKORO_LANG_MAP = {
    'en-us': 'a', 'en': 'a',
    'en-gb': 'b', 'en-uk': 'b',
    'fr': 'f', 'fr-fr': 'f',
    'ja': 'j', 'jp': 'j',
    'zh': 'z', 'cn': 'z',
    'es': 'e',
    'it': 'i',
    'pt': 'p', 'pt-br': 'p',
    'hi': 'h',
    'de': 'd',
    'ru': 'r',
    'tr': 't'
}

def map_lang_code(lang_code: str) -> str:
    """
    Maps a standard language code (e.g., 'en', 'es') to Kokoro-internal code (e.g., 'a', 'e').
    Returns the original code if no mapping is found.
    """
    if not lang_code:
        return 'a'
    return KOKORO_LANG_MAP.get(lang_code.lower(), lang_code)
