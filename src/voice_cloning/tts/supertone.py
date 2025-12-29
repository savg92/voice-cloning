"""
Supertone Supertonic TTS Wrapper
Lightning-fast, on-device TTS using ONNX Runtime
Based on official implementation from https://github.com/supertone-inc/supertonic
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Tuple
from unicodedata import normalize

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class UnicodeProcessor:
    """Text processor for converting text to character IDs."""
    
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)
    
    def _preprocess_text(self, text: str) -> str:
        """Normalize and clean text."""
        text = normalize("NFKD", text)
        
        # Remove emojis
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
        
        # Character replacements
        replacements = {
            "–": "-", "‑": "-", "—": "-", "¯": " ", "_": " ",
            """: '"', """: '"', "'": "'", "´": "'", "`": "'",
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
        
        return text
    
    def __call__(self, text_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert list of texts to text_ids and masks."""
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        
        for i, text in enumerate(text_list):
            unicode_vals = np.array([ord(char) for char in text], dtype=np.uint16)
            text_ids[i, :len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        
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


class SupertoneTTS:
    """
    Supertone Supertonic TTS wrapper.
    Ultra-fast on-device TTS with ONNX Runtime.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize Supertonic TTS.
        
        Args:
            model_dir: Directory containing ONNX models and voice styles.
        """
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install: uv pip install onnxruntime"
            )
        
        # Set directories
        if model_dir is None:
            model_dir = Path("models/supertonic")
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.onnx_dir = model_dir / "onnx"
        self.voice_styles_dir = model_dir / "voice_styles"
        
        if not self.onnx_dir.exists():
            raise FileNotFoundError(
                f"ONNX directory not found: {self.onnx_dir}\n"
                "Download models: git clone https://huggingface.co/Supertone/supertonic models/supertonic"
            )
        
        # Load config
        cfg_path = self.onnx_dir / "tts.json"
        with open(cfg_path, 'r') as f:
            self.cfgs = json.load(f)
        
        # Initialize text processor
        unicode_indexer_path = self.onnx_dir / "unicode_indexer.json"
        self.text_processor = UnicodeProcessor(str(unicode_indexer_path))
        
        # Load ONNX models
        logger.info("Loading Supertonic ONNX models...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']
        
        self.dp_ort = ort.InferenceSession(
            str(self.onnx_dir / "duration_predictor.onnx"),
            sess_options=sess_options, providers=providers
        )
        self.text_enc_ort = ort.InferenceSession(
            str(self.onnx_dir / "text_encoder.onnx"),
            sess_options=sess_options, providers=providers
        )
        self.vector_est_ort = ort.InferenceSession(
            str(self.onnx_dir / "vector_estimator.onnx"),
            sess_options=sess_options, providers=providers
        )
        self.vocoder_ort = ort.InferenceSession(
            str(self.onnx_dir / "vocoder.onnx"),
            sess_options=sess_options, providers=providers
        )
        
        # Model params
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]
        
        logger.info("✓ Supertonic models loaded successfully")
    
    def load_voice_style(self, voice_style_name: str) -> Style:
        """Load voice style from JSON file."""
        style_path = self.voice_styles_dir / f"{voice_style_name}.json"
        
        if not style_path.exists():
            logger.warning(f"Style not found: {voice_style_name}, using F1")
            style_path = self.voice_styles_dir / "F1.json"
        
        with open(style_path, "r") as f:
            voice_style = json.load(f)
        
        ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
        ttl_dims = voice_style["style_ttl"]["dims"]
        ttl_style = ttl_data.reshape(ttl_dims[1], ttl_dims[2])
        
        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
        dp_dims = voice_style["style_dp"]["dims"]
        dp_style = dp_data.reshape(dp_dims[1], dp_dims[2])
        
        # Add batch dimension
        ttl_style = np.expand_dims(ttl_style, axis=0)
        dp_style = np.expand_dims(dp_style, axis=0)
        
        return Style(ttl_style, dp_style)
    
    def list_voice_styles(self) -> List[str]:
        """List available voice styles."""
        if not self.voice_styles_dir.exists():
            return []
        return [f.stem for f in self.voice_styles_dir.glob("*.json")]
    
    def _sample_noisy_latent(self, duration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample initial noisy latent for diffusion."""
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        latent_dim = self.ldim * self.chunk_compress_factor
        
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        
        # Create latent mask
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_lengths = (wav_lengths + latent_size - 1) // latent_size
        max_len = latent_lengths.max()
        ids = np.arange(0, max_len)
        latent_mask = (ids < np.expand_dims(latent_lengths, axis=1)).astype(np.float32)
        latent_mask = latent_mask.reshape(-1, 1, max_len)
        
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_style: str = "F1",
        steps: int = 8,
        speed: float = 1.0,
        stream: bool = False,
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            output_path: Output WAV file path
            voice_style: Voice style name (F1, F2, M1, M2)
            steps: Inference steps (higher = better quality)
            speed: Speech speed (higher = faster, default: 1.0)
            stream: Enable pseudo-streaming (play sentences as they generate)
            
        Returns:
            Path to output file
        """
        # Load style
        style = self.load_voice_style(voice_style)
        
        if stream:
            import threading
            import queue
            import subprocess
            import tempfile
            import re
            
            # Simple sentence splitter
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            playback_queue = queue.Queue()
            stop_event = threading.Event()
            
            def playback_worker():
                while not stop_event.is_set() or not playback_queue.empty():
                    try:
                        audio_file = playback_queue.get(timeout=0.1)
                        if audio_file is None:
                            break
                        
                        # Play audio (macOS specific for now, fallback to ffplay)
                        try:
                            subprocess.run(["afplay", audio_file], check=True)
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            try:
                                subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file], 
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            except Exception:
                                pass
                                
                        # Cleanup
                        try:
                            os.remove(audio_file)
                        except Exception:
                            pass
                            
                        playback_queue.task_done()
                    except queue.Empty:
                        continue
            
            # Start playback thread
            player_thread = threading.Thread(target=playback_worker, daemon=True)
            player_thread.start()
            
            full_audio_parts = []
            
            logger.info("Starting streaming generation...")
            for i, sentence in enumerate(sentences):
                # Generate audio for sentence
                # Reuse internal logic but skip file saving for now
                # Actually, reusing the full pipeline is easier if we refactor, 
                # but for minimal changes, let's just copy-paste the core logic or extract it.
                # Extracting core logic is cleaner.
                
                wav = self._generate_waveform(sentence, style, steps, speed)
                full_audio_parts.append(wav)
                
                # Save chunk to temp file for playback
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, wav, self.sample_rate)
                    playback_queue.put(tmp.name)
            
            # Wait for playback to finish
            stop_event.set()
            player_thread.join()
            
            # Combine all parts for final output
            final_wav = np.concatenate(full_audio_parts)
            sf.write(output_path, final_wav, self.sample_rate)
            logger.info(f"✓ Audio saved to: {output_path}")
            
            return output_path

        else:
            # Normal full generation
            wav = self._generate_waveform(text, style, steps, speed)
            sf.write(output_path, wav, self.sample_rate)
            logger.info(f"✓ Audio saved to: {output_path}")
            return output_path

    def _generate_waveform(self, text: str, style: Style, steps: int, speed: float) -> np.ndarray:
        """Internal method to generate waveform from text."""
        # Process text
        text_ids, text_mask = self.text_processor([text])
        
        # Duration prediction
        dur, *_ = self.dp_ort.run(
            None,
            {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur = dur / speed
        
        # Text encoding
        text_emb, *_ = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
        )
        
        # Sample initial noise
        xt, latent_mask = self._sample_noisy_latent(dur)
        
        # Iterative denoising
        bsz = 1
        total_step_np = np.array([steps] * bsz, dtype=np.float32)
        
        for step in range(steps):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                }
            )
        
        # Vocoder
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        
        # Trim to actual duration
        duration_samples = int(self.sample_rate * dur[0])
        wav = wav[0, :duration_samples]
        
        return wav


def synthesize_with_supertone(
    text: str,
    output_path: str,
    model_dir: Optional[str] = None,
    preset: Optional[str] = None,
    steps: int = 8,
    cfg_scale: float = 1.0,
    stream: bool = False,
) -> str:
    """
    Convenience function to synthesize speech.
    
    Args:
        text: Text to synthesize
        output_path: Output file path
        model_dir: Model directory (optional)
        preset: Voice style (F1, F2, M1, M2)
        steps: Inference steps
        cfg_scale: Unused (kept for compatibility)
        stream: Enable streaming playback
    
    Returns:
        Path to output file
    """
    tts = SupertoneTTS(model_dir=model_dir)
    
    if preset is None:
        preset = "F1"
    
    return tts.synthesize(
        text=text,
        output_path=output_path,
        voice_style=preset,
        steps=steps,
        speed=1.0,
        stream=stream
    )

