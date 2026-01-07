import logging
import sys
import numpy as np
from voice_cloning.tts.supertonic2 import Supertonic2TTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('supertonic_debug.log')
    ]
)
logger = logging.getLogger("debug_script")

def debug_synthesis():
    tts = Supertonic2TTS(use_cpu=True)
    
    text = "Hello world"
    logger.info(f"Processing text: '{text}'")
    
    # Manually call internal methods to inspect
    text_ids, text_mask = tts.text_processor([text])
    logger.info(f"Text IDs: {text_ids}")
    logger.info(f"Text Mask: {text_mask}")
    
    # Check for 0s (unknowns)
    if 0 in text_ids:
        logger.warning("Found unknown characters (mapped to 0) in Text IDs!")
    
    # Load style
    style = tts.load_voice_style("F1")
    
    # Run duration predictor
    dur, *_ = tts.dp_ort.run(
        None,
        {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    )
    logger.info(f"Predicted Duration (raw): {dur}")
    
    # Generate wav
    wav = tts.synthesize(text, "debug_output.wav", steps=10, use_cpu=True)
    logger.info(f"Generated WAV at {wav}")

if __name__ == "__main__":
    debug_synthesis()
