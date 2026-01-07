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
    
    def _preprocess_text(self, text: str, lang: str) -> str:
        """Normalize and clean text."""
        # Use NFKD to decompose characters (needed for Korean Jamo)
        text = normalize("NFKD", text)
        
        # Remove emojis (wide Unicode range)
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
        
        # Add language tags
        text = f"<{lang}>{text}</{lang}>"
        
        return text
    
    def __call__(self, text_list: list[str], lang_list: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Convert list of texts to text_ids and masks."""
        if lang_list is None:
             lang_list = ["en"] * len(text_list)
             
        text_list = [self._preprocess_text(t, l) for t, l in zip(text_list, lang_list)]
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
    
    # Supported languages
    SUPPORTED_LANGUAGES = {'en', 'ko', 'es', 'pt', 'fr'}
    
    # Language name mapping for error messages
    LANGUAGE_NAMES = {
        'en': 'English',
        'ko': 'Korean',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'fr': 'French'
    }
    
    # Voice style cache (class-level to share across instances)
    _voice_style_cache = {}
    
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
        
        # Enable threading optimizations for CPU
        sess_options.inter_op_num_threads = 0  # Auto-select
        sess_options.intra_op_num_threads = 0  # Auto-select
        
        # Smart execution provider selection
        providers = self._get_optimal_providers(use_cpu)
        
        logger.info(f"Using execution providers: {providers}")
        
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
    
    def _get_optimal_providers(self, use_cpu: bool) -> list[str]:
        """Intelligently select the best available execution providers."""
        if use_cpu:
            return ['CPUExecutionProvider']
        
        available = self.ort.get_available_providers()
        providers = []
        
        # Priority: TensorRT > CUDA > DirectML > ROCm > CoreML > CPU
        # TensorRT (NVIDIA - best performance)
        if 'TensorrtExecutionProvider' in available:
            providers.append('TensorrtExecutionProvider')
            logger.info("Using TensorRT for optimal NVIDIA GPU performance")
        
        # CUDA (NVIDIA - good general performance)
        elif 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
            logger.info("Using CUDA for NVIDIA GPU acceleration")
        
        # DirectML (Windows - cross-vendor GPU)
        elif 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
            logger.info("Using DirectML for Windows GPU acceleration")
        
        # ROCm (AMD GPU)
        elif 'ROCMExecutionProvider' in available:
            providers.append('ROCMExecutionProvider')
            logger.info("Using ROCm for AMD GPU acceleration")
        
        # CoreML (Apple Silicon/macOS)
        elif 'CoreMLExecutionProvider' in available and sys.platform == "darwin":
            providers.append('CoreMLExecutionProvider')
            logger.info("Using CoreML for Apple Silicon acceleration")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        if len(providers) == 1:
            logger.warning("No GPU acceleration available, using CPU only. This may be slower.")
        
        return providers
    
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
        """Load voice style from JSON file with caching."""
        # Check cache first
        cache_key = f"{self.voice_styles_dir}/{voice_style_name}"
        if cache_key in self._voice_style_cache:
            return self._voice_style_cache[cache_key]
        
        style_path = self.voice_styles_dir / f"{voice_style_name}.json"
        
        if not style_path.exists():
            # Try to find any available style
            available_styles = self.list_voice_styles()
            if available_styles:
                logger.warning(
                    f"Voice style '{voice_style_name}' not found. "
                    f"Using '{available_styles[0]}' instead. "
                    f"Available styles: {', '.join(available_styles)}"
                )
                voice_style_name = available_styles[0]
                style_path = self.voice_styles_dir / f"{voice_style_name}.json"
                cache_key = f"{self.voice_styles_dir}/{voice_style_name}"
            else:
                raise FileNotFoundError(
                    f"No voice styles found in {self.voice_styles_dir}. "
                    f"Please check that the model files were downloaded correctly."
                )
        
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
        
        # Create and cache the style
        style = Style(ttl_style, dp_style)
        self._voice_style_cache[cache_key] = style
        
        return style
    
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
    
    def _apply_pitch_shift(self, wav: np.ndarray, semitones: float) -> np.ndarray:
        """Apply pitch shifting to audio using librosa."""
        if semitones == 0:
            return wav
        
        try:
            import librosa
            return librosa.effects.pitch_shift(
                wav, sr=self.sample_rate, n_steps=semitones
            )
        except ImportError:
            logger.warning("librosa not available, skipping pitch shift")
            return wav
    
    def _apply_energy_scale(self, wav: np.ndarray, scale: float) -> np.ndarray:
        """Scale audio energy/amplitude."""
        if scale == 1.0:
            return wav
        
        scaled = wav * scale
        
        # Prevent clipping
        max_val = np.abs(scaled).max()
        if max_val > 1.0:
            scaled = scaled / max_val
            logger.warning(f"Energy scaling caused clipping, normalized to prevent distortion")
        
        return scaled
    
    def _crossfade_audio(self, audio1: np.ndarray, audio2: np.ndarray, overlap_samples: int) -> np.ndarray:
        """Crossfade two audio segments for smooth transitions."""
        if overlap_samples <= 0 or len(audio1) < overlap_samples:
            # No overlap, just concatenate
            return np.concatenate([audio1, audio2])
        
        # Create fade curves using Hann window
        fade_out = np.hanning(overlap_samples * 2)[:overlap_samples]
        fade_in = np.hanning(overlap_samples * 2)[overlap_samples:]
        
        # Split audio1 into main part and tail
        main1 = audio1[:-overlap_samples]
        tail1 = audio1[-overlap_samples:]
        
        # Get head of audio2
        if len(audio2) < overlap_samples:
            # audio2 is shorter than overlap, adjust
            actual_overlap = len(audio2)
            fade_out = fade_out[:actual_overlap]
            fade_in = fade_in[:actual_overlap]
            tail1 = tail1[-actual_overlap:]
            head2 = audio2
            rest2 = np.array([])
        else:
            head2 = audio2[:overlap_samples]
            rest2 = audio2[overlap_samples:]
        
        # Apply crossfade
        crossfaded = tail1 * fade_out + head2 * fade_in
        
        # Concatenate: main1 + crossfaded + rest2
        return np.concatenate([main1, crossfaded, rest2])
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_style: str | None = None,
        lang_code: str = "en",
        steps: int = 8,
        speed: float = 1.0,
        stream: bool = False,
        pitch_shift: float = 0.0,
        energy_scale: float = 1.0,
        use_cpu: bool = False  # Kept for API compatibility, but handled in init
    ) -> str:
        Synthesize speech from text.
        
        Args:
            text: Input text
            output_path: Output WAV file path
            voice_style: Voice style name (if None, use first available)
            lang_code: Language code (en, ko, es, pt, fr)
            steps: Inference steps (higher = better quality)
            speed: Speech speed (higher = faster, default: 1.0)
            stream: Enable pseudo-streaming (play sentences as they generate)
            pitch_shift: Pitch shift in semitones (-12 to +12)
            energy_scale: Energy/amplitude scale (0.5 to 2.0)
            
        Returns:
            Path to output file
        """
        # Input validation
        # Validate text
        if not text or not text.strip():
            raise ValueError("Text cannot be empty. Please provide text to synthesize.")
        
        # Validate language code
        if lang_code not in self.SUPPORTED_LANGUAGES:
            lang_names = ', '.join(f"{code} ({self.LANGUAGE_NAMES[code]})" 
                                   for code in sorted(self.SUPPORTED_LANGUAGES))
            raise ValueError(
                f"Unsupported language code: '{lang_code}'. "
                f"Supported languages: {lang_names}"
            )
        
        # Validate steps
        if not (1 <= steps <= 50):
            raise ValueError(
                f"Steps must be between 1 and 50, got {steps}. "
                f"Recommended: 8-10 for speed, 20-30 for quality."
            )
        
        # Validate speed
        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}.")
        
        if speed < 0.5 or speed > 2.0:
            logger.warning(
                f"Speed {speed} is outside the recommended range (0.5-2.0). "
                f"Audio quality may be affected."
            )
        
        # Get voice style
        if voice_style is None:
            styles = self.list_voice_styles()
            if not styles:
                raise RuntimeError(
                    f"No voice styles found in {self.voice_styles_dir}. "
                    f"Please ensure the model was downloaded correctly."
                )
            voice_style = styles[0]
            logger.info(f"No voice style specified, using default: {voice_style}")

        # Load style (with caching)
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
            
            # Calculate overlap for crossfading (50ms)
            overlap_samples = int(0.05 * self.sample_rate)
            
            full_audio_parts = []
            
            logger.info("Starting streaming generation with crossfade...")
            for i, sentence in enumerate(sentences):
                wav = self._generate_waveform(sentence, style, steps, speed, lang_code=lang_code)
                
                # Apply prosody controls
                if pitch_shift != 0.0:
                    wav = self._apply_pitch_shift(wav, pitch_shift)
                if energy_scale != 1.0:
                    wav = self._apply_energy_scale(wav, energy_scale)
                
                # Crossfade with previous chunk for smooth transitions
                if i > 0 and len(full_audio_parts) > 0:
                    # Combine previous with current using crossfade
                    prev_wav = full_audio_parts[-1]
                    combined = self._crossfade_audio(prev_wav, wav, overlap_samples)
                    # Replace last chunk with combined
                    full_audio_parts[-1] = combined
                else:
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
            wav = self._generate_waveform(text, style, steps, speed, lang_code=lang_code)
            
            # Apply prosody controls
            if pitch_shift != 0.0:
                wav = self._apply_pitch_shift(wav, pitch_shift)
            if energy_scale != 1.0:
                wav = self._apply_energy_scale(wav, energy_scale)
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(output_path, wav, self.sample_rate)
            logger.info(f"✓ Audio saved to: {output_path}")
            return str(output_path)

    def _generate_waveform(self, text: str, style: Style, steps: int, speed: float, lang_code: str = "en") -> np.ndarray:
        """Internal method to generate waveform from text."""
        # Process text
        text_ids, text_mask = self.text_processor([text], [lang_code])
        
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
        
        # Normalize audio to prevent clipping/distortion
        max_val = np.abs(wav).max()
        if max_val > 1.0:
            wav = wav / max_val
        
        return wav
    
    def synthesize_batch(
        self,
        texts: list[str],
        output_paths: list[str] | None = None,
        voice_style: str | None = None,
        lang_codes: list[str] | None = None,
        steps: int = 8,
        speed: float = 1.0,
        combine_output: bool = False,
        output_path: str | None = None
    ) -> list[str] | str:
        """
        Batch synthesize multiple texts efficiently.
        
        Args:
            texts: List of input texts
            output_paths: List of output paths (if None, auto-generate)
            voice_style: Voice style name (applied to all)
            lang_codes: Language codes per text (if None, all use 'en')
            steps: Inference steps
            speed: Speech speed
            combine_output: If True, concatenate all audio into single file
            output_path: Path for combined output (required if combine_output=True)
            
        Returns:
            List of output paths, or single path if combine_output=True
        """
        # Validation
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        if combine_output and not output_path:
            raise ValueError("output_path required when combine_output=True")
        
        batch_size = len(texts)
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
        
        # Setup language codes
        if lang_codes is None:
            lang_codes = ["en"] * batch_size
        elif len(lang_codes) != batch_size:
            raise ValueError(f"lang_codes length ({len(lang_codes)}) must match texts length ({batch_size})")
        
        # Validate all language codes
        for i, lang_code in enumerate(lang_codes):
            if lang_code not in self.SUPPORTED_LANGUAGES:
                lang_names = ', '.join(f"{code} ({self.LANGUAGE_NAMES[code]})" 
                                       for code in sorted(self.SUPPORTED_LANGUAGES))
                raise ValueError(
                    f"Unsupported language code '{lang_code}' at index {i}. "
                    f"Supported: {lang_names}"
                )
        
        # Validate steps and speed (same as single)
        if not (1 <= steps <= 50):
            raise ValueError(f"Steps must be between 1 and 50, got {steps}")
        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}")
        
        # Get voice style
        if voice_style is None:
            styles = self.list_voice_styles()
            if not styles:
                raise RuntimeError(f"No voice styles found in {self.voice_styles_dir}")
            voice_style = styles[0]
            logger.info(f"Using default voice style: {voice_style}")
        
        # Load style (with caching)
        style = self.load_voice_style(voice_style)
        
        # Generate output paths if not provided
        if output_paths is None and not combine_output:
            import tempfile
            output_paths = [tempfile.mktemp(suffix=".wav") for _ in range(batch_size)]
        elif output_paths and len(output_paths) != batch_size:
            raise ValueError(f"output_paths length ({len(output_paths)}) must match texts length ({batch_size})")
        
        logger.info(f"Batch synthesizing {batch_size} texts...")
        
        # Generate waveforms for all texts
        waveforms = []
        for i, (text, lang_code) in enumerate(zip(texts, lang_codes)):
            wav = self._generate_waveform(text, style, steps, speed, lang_code=lang_code)
            waveforms.append(wav)
        
        if combine_output:
            # Concatenate all waveforms
            combined_wav = np.concatenate(waveforms)
            
            # Save combined output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, combined_wav, self.sample_rate)
            logger.info(f"✓ Combined audio saved to: {output_path}")
            
            return str(output_path)
        else:
            # Save individual outputs
            for i, (wav, out_path) in enumerate(zip(waveforms, output_paths)):
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(out_path, wav, self.sample_rate)
            
            logger.info(f"✓ Batch synthesis complete: {batch_size} files")
            return [str(p) for p in output_paths]


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
