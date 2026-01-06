"""
Supertone Supertonic-2 TTS Wrapper
Lightning-fast, on-device Multilingual TTS using ONNX Runtime
Based on Supertone/supertonic-2
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from unicodedata import normalize

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class UnicodeProcessor:
    """Text processor for converting text to character IDs."""
    
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path) as f:
            self.indexer = json.load(f)
    
    def _preprocess_text(self, text: str) -> str:
        """Normalize and clean text."""
        # Use NFKD to decompose characters (needed for Korean Jamo)
        # The unicode_indexer supports Jamo (e.g. 4352) but not Syllables (e.g. 44032)
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
    
    def __call__(self, text_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Convert list of texts to text_ids and masks."""
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        
        for i, text in enumerate(text_list):
            unicode_vals = np.array([ord(char) for char in text], dtype=np.uint16)
            
            # Check for unknown chars
            mapped_vals = []
            for char_val in unicode_vals:
                if char_val >= len(self.indexer):
                    logger.warning(f"Character {chr(char_val)} ({char_val}) out of indexer range")
                    mapped_vals.append(0)
                else:
                    idx = self.indexer[char_val]
                    if idx == -1:
                        # logger.warning(f"Character {chr(char_val)} ({char_val}) mapped to unknown (-1)")
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


class Supertonic2TTS:
    """
    Supertone Supertonic-2 TTS wrapper.
    Ultra-fast on-device Multilingual TTS with ONNX Runtime.
    """
    
    def __init__(self, model_dir: str | None = None, use_cpu: bool = False):
        """
        Initialize Supertonic-2 TTS.
        
        Args:
            model_dir: Directory containing ONNX models and voice styles.
            use_cpu: Force CPU execution (disable CoreML/others).
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
            model_dir = Path("models/supertonic2")
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.onnx_dir = model_dir # v2 seems to put things in root or 'onnx'
        self.voice_styles_dir = model_dir / "voice_styles"
        
        # Hybrid download approach
        if not (self.model_dir / "tts.json").exists():
            logger.info(f"Models not found in {self.model_dir}. Downloading from Hugging Face...")
            self.download_models()
        
        # If still not found (e.g. download failed), try 'onnx' subdir as in v1
        if not (self.model_dir / "tts.json").exists() and (self.model_dir / "onnx" / "tts.json").exists():
             self.onnx_dir = model_dir / "onnx"
        
        # Load config
        cfg_path = self.onnx_dir / "tts.json"
        with open(cfg_path) as f:
            self.cfgs = json.load(f)
        
        # Initialize text processor
        unicode_indexer_path = self.onnx_dir / "unicode_indexer.json"
        self.text_processor = UnicodeProcessor(str(unicode_indexer_path))
        
        # Load ONNX models
        logger.info("Loading Supertonic-2 ONNX models...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Prefer CoreML on macOS if available, otherwise CPU
        providers = ['CPUExecutionProvider']
        if not use_cpu and sys.platform == "darwin":
            try:
                # Check if CoreML is available
                if 'CoreMLExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CoreMLExecutionProvider')
            except Exception:
                pass
        
        logger.info(f"Using providers: {providers}")
        
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
        
        logger.info("✓ Supertonic-2 models loaded successfully")
    
    def download_models(self):
        """Download Supertonic-2 models from Hugging Face."""
        try:
            snapshot_download(
                repo_id="Supertone/supertonic-2",
                local_dir=str(self.model_dir),
                allow_patterns=["*.onnx", "*.json", "voice_styles/*.json"]
            )
            logger.info("✓ Download complete")
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            raise
            
    def load_voice_style(self, voice_style_name: str) -> Style:
        """Load voice style from JSON file."""
        style_path = self.voice_styles_dir / f"{voice_style_name}.json"
        
        if not style_path.exists():
            # Try to find any available style
            available_styles = self.list_voice_styles()
            if available_styles:
                logger.warning(f"Style not found: {voice_style_name}, using {available_styles[0]}")
                style_path = self.voice_styles_dir / f"{available_styles[0]}.json"
            else:
                raise FileNotFoundError(f"No voice styles found in {self.voice_styles_dir}")
        
        with open(style_path) as f:
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
    
    def list_voice_styles(self) -> list[str]:
        """List available voice styles."""
        if not self.voice_styles_dir.exists():
            return []
        return [f.stem for f in self.voice_styles_dir.glob("*.json")]
    
    def _sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        voice_style: str | None = None,
        lang_code: str = "en",
        steps: int = 8,
        speed: float = 1.0,
        stream: bool = False,
        use_cpu: bool = False # Kept for API compatibility, but handled in init
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            output_path: Output WAV file path
            voice_style: Voice style name (if None, use first available)
            lang_code: Language code (en, ko, es, pt, fr)
            steps: Inference steps (higher = better quality)
            speed: Speech speed (higher = faster, default: 1.0)
            stream: Enable pseudo-streaming (play sentences as they generate)
            
        Returns:
            Path to output file
        """
        if voice_style is None:
            styles = self.list_voice_styles()
            voice_style = styles[0] if styles else "F1"

        # Load style
        style = self.load_voice_style(voice_style)
        
        if stream:
            import threading
            import queue
            import subprocess
            import tempfile
            
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
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(output_path, final_wav, self.sample_rate)
            logger.info(f"✓ Audio saved to: {output_path}")
            
            return str(output_path)

        else:
            wav = self._generate_waveform(text, style, steps, speed)
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(output_path, wav, self.sample_rate)
            logger.info(f"✓ Audio saved to: {output_path}")
            return str(output_path)

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
            
            # The model predicts the vector field (velocity) v_t
            vt, *_ = self.vector_est_ort.run(
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
            
            # Euler integration: x_{t+1} = x_t + v_t * dt
            # Flow Matching typically goes from Noise (t=0) to Data (t=1)
            dt = 1.0 / steps
            xt = xt + vt * dt
        
        # Vocoder
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        
        # Trim to actual duration
        duration_samples = int(self.sample_rate * dur[0])
        wav = wav[0, :duration_samples]
        
        # Normalize audio to prevent clipping/distortion
        max_val = np.abs(wav).max()
        if max_val > 1.0:
            wav = wav / max_val
        
        return wav


def synthesize_speech(
    text: str,
    output_path: str,
    model_dir: str | None = None,
    voice: str | None = None,
    lang_code: str = "en",
    steps: int = 8,
    speed: float = 1.0,
    stream: bool = False,
    use_cpu: bool = False,
    **kwargs
) -> str:
    """
    Convenience function for Supertonic-2 TTS.
    """
    tts = Supertonic2TTS(model_dir=model_dir, use_cpu=use_cpu)
    return tts.synthesize(
        text=text,
        output_path=output_path,
        voice_style=voice,
        lang_code=lang_code,
        steps=steps,
        speed=speed,
        stream=stream
    )
