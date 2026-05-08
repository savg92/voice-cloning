"""
Supertone Supertonic-3 TTS Wrapper
Lightning-fast, on-device Multilingual TTS using ONNX Runtime.
Based on Supertone/supertonic-3.
Supports 31 languages and expression tags (<laugh>, <breath>, <sigh>).
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from unicodedata import normalize
from typing import Optional, Union, List, Tuple

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class UnicodeProcessor:
    """Text processor for converting text to character IDs."""
    
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path) as f:
            self.indexer = json.load(f)
    
    def _preprocess_text(self, text: str, lang: str) -> str:
        """Normalize and clean text."""
        # Use NFKD to decompose characters
        text = normalize("NFKD", text)
        
        # Character replacements
        replacements = {
            "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
            "“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
            "→": " ", "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        # Expression replacements
        text = text.replace("@", " at ")
        text = text.replace("e.g.,", "for example, ")
        text = text.replace("i.e.,", "that is, ")
        
        # Fix spacing
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        
        # Remove duplicate spaces
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        # Add language tags
        text = f"<{lang}>{text}</{lang}>"
        
        return text
    
    def __call__(self, text_list: List[str], lang_list: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert list of texts to text_ids and masks."""
        if lang_list is None:
             lang_list = ["en"] * len(text_list)
             
        text_list = [self._preprocess_text(t, l) for t, l in zip(text_list, lang_list)]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        
        for i, text in enumerate(text_list):
            unicode_vals = np.array([ord(char) for char in text], dtype=np.uint16)
            
            mapped_vals = []
            for char_val in unicode_vals:
                if char_val >= len(self.indexer):
                    logger.warning(f"Character {chr(char_val)} ({char_val}) out of indexer range")
                    mapped_vals.append(0)
                else:
                    idx = self.indexer[char_val]
                    if idx == -1:
                        mapped_vals.append(0)
                    else:
                        mapped_vals.append(idx)
                        
            text_ids[i, :len(unicode_vals)] = np.array(mapped_vals, dtype=np.int64)
        
        # Create mask
        max_len = text_ids_lengths.max()
        ids = np.arange(0, max_len)
        text_mask = (ids < np.expand_dims(text_ids_lengths, axis=1)).astype(np.float32)
        text_mask = text_mask.reshape(-1, 1, max_len)
        
        return text_ids, text_mask

class Style:
    """Voice style container."""
    def __init__(self, style_ttl: np.ndarray, style_dp: np.ndarray):
        self.ttl = style_ttl
        self.dp = style_dp

class Supertonic3TTS:
    """
    Supertone Supertonic-3 TTS wrapper.
    Ultra-fast on-device Multilingual TTS with ONNX Runtime.
    """
    
    # Supported languages (31)
    SUPPORTED_LANGUAGES = {
        'en', 'ko', 'ja', 'zh', 'es', 'fr', 'pt', 'de', 'it', 'ru', 
        'tr', 'vi', 'pl', 'nl', 'ar', 'hi', 'sv', 'da', 'fi', 'nb', 
        'cs', 'el', 'hu', 'ro', 'uk', 'id', 'ms', 'th', 'he', 'fa', 'ca'
    }
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, use_cpu: bool = False):
        """
        Initialize Supertonic-3 TTS.
        """
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install: uv add onnxruntime"
            )
        
        if model_dir is None:
            model_dir = Path("models/supertonic3")
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        
        # Download models if missing
        if not (self.model_dir / "tts.json").exists() and not (self.model_dir / "onnx" / "tts.json").exists():
            self.download_models()
            
        self.onnx_dir = self.model_dir if (self.model_dir / "tts.json").exists() else self.model_dir / "onnx"
        self.voice_styles_dir = self.model_dir / "voice_styles"
        
        # Load config
        with open(self.onnx_dir / "tts.json") as f:
            self.cfgs = json.load(f)
            
        # Initialize text processor
        unicode_indexer_path = self.onnx_dir / "unicode_indexer.json"
        self.text_processor = UnicodeProcessor(str(unicode_indexer_path))
        
        # Load ONNX models
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = self._get_optimal_providers(use_cpu)
        
        self.dp_ort = ort.InferenceSession(str(self.onnx_dir / "duration_predictor.onnx"), sess_options, providers)
        self.text_enc_ort = ort.InferenceSession(str(self.onnx_dir / "text_encoder.onnx"), sess_options, providers)
        self.vector_est_ort = ort.InferenceSession(str(self.onnx_dir / "vector_estimator.onnx"), sess_options, providers)
        self.vocoder_ort = ort.InferenceSession(str(self.onnx_dir / "vocoder.onnx"), sess_options, providers)
        
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]
        
        logger.info("✓ Supertonic-3 models loaded successfully")

    def _get_optimal_providers(self, use_cpu: bool) -> List[str]:
        if use_cpu: return ['CPUExecutionProvider']
        available = self.ort.get_available_providers()
        providers = []
        if 'TensorrtExecutionProvider' in available: providers.append('TensorrtExecutionProvider')
        elif 'CUDAExecutionProvider' in available: providers.append('CUDAExecutionProvider')
        elif 'CoreMLExecutionProvider' in available and sys.platform == "darwin": providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')
        return providers

    def download_models(self):
        logger.info(f"Downloading Supertonic-3 models to {self.model_dir}...")
        snapshot_download(
            repo_id="Supertone/supertonic-3",
            local_dir=str(self.model_dir),
            allow_patterns=["*.onnx", "*.json", "voice_styles/*.json"]
        )

    def load_voice_style(self, voice_style_name: str) -> Style:
        style_path = self.voice_styles_dir / f"{voice_style_name}.json"
        if not style_path.exists():
            # Fallback to first available
            styles = [f.stem for f in self.voice_styles_dir.glob("*.json")]
            if not styles: raise FileNotFoundError(f"No styles in {self.voice_styles_dir}")
            style_path = self.voice_styles_dir / f"{styles[0]}.json"
            
        with open(style_path) as f:
            voice_style = json.load(f)
            
        ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).reshape(1, *voice_style["style_ttl"]["dims"][1:])
        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).reshape(1, *voice_style["style_dp"]["dims"][1:])
        return Style(ttl_data, dp_data)

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = "F1",
        lang: str = "en",
        speed: float = 1.0,
        steps: int = 8,
    ) -> str:
        style = self.load_voice_style(voice)
        text_ids, text_mask = self.text_processor([text], [lang])
        
        # Inference
        dur, *_ = self.dp_ort.run(None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask})
        dur = dur / speed
        text_emb, *_ = self.text_enc_ort.run(None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask})
        
        # Sampling
        wav_lengths = (dur * self.sample_rate).astype(np.int64)
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int((wav_lengths.max() + latent_size - 1) / latent_size)
        latent_dim = self.ldim * self.chunk_compress_factor
        
        xt = np.random.randn(1, latent_dim, latent_len).astype(np.float32)
        latent_mask = (np.arange(latent_len) < ((wav_lengths + latent_size - 1) // latent_size)).astype(np.float32).reshape(1, 1, -1)
        xt *= latent_mask
        
        total_step = np.array([steps], dtype=np.float32)
        for step in range(steps):
            current_step = np.array([step], dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(None, {
                "noisy_latent": xt, "text_emb": text_emb, "style_ttl": style.ttl,
                "text_mask": text_mask, "latent_mask": latent_mask,
                "current_step": current_step, "total_step": total_step
            })
            
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        wav = wav[0, :int(self.sample_rate * dur[0])]
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, wav, self.sample_rate)
        return str(output_path)

def synthesize_with_supertonic3(
    text: str,
    output_path: str,
    model_dir: Optional[str] = None,
    preset: str = "F1",
    lang_code: str = "en",
    speed: float = 1.0,
    steps: int = 8,
    **kwargs
) -> str:
    if lang_code == 'e': lang_code = 'en'
    tts = Supertonic3TTS(model_dir=model_dir)
    return tts.synthesize(text, output_path, voice=preset, lang=lang_code, speed=speed, steps=steps)
