import torch
import logging

logger = logging.getLogger(__name__)

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
    'hi': 'h'
}

def map_lang_code(lang_code: str) -> str:
    """
    Maps a standard language code (e.g., 'en', 'es') to Kokoro-internal code (e.g., 'a', 'e').
    Returns the original code if no mapping is found.
    """
    if not lang_code:
        return 'a'
    return KOKORO_LANG_MAP.get(lang_code.lower(), lang_code)

def get_best_device() -> str:
    """
    Detects and returns the best available device.
    Priority: CUDA > MPS > XLA > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # Check for TPU (XLA)
    try:
        import torch_xla.core.xla_model as xm
        # This will return something like 'xla:0'
        return str(xm.xla_device())
    except ImportError:
        pass
        
    return "cpu"
