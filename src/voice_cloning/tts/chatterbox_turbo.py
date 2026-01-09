import torch
import torchaudio as ta
import shutil
import threading
import queue
import subprocess
import numpy as np
import soundfile as sf
import os
import logging
from .utils import map_lang_code
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Voice Presets
VOICE_PRESETS = {
    "en": "samples/kokoro_voices/af_heart.wav",
    "es": "samples/kokoro_voices/ef_dora.wav",
    "fr": "samples/kokoro_voices/ff_siwis.wav",
    "it": "samples/kokoro_voices/if_sara.wav",
    "pt": "samples/kokoro_voices/pf_dora.wav",
    "de": "samples/kokoro_voices/de_sarah.wav",
    "ru": "samples/kokoro_voices/ru_nicole.wav",
    "tr": "samples/kokoro_voices/tr_river.wav",
    "hi": "samples/kokoro_voices/hf_alpha.wav",
    "af_heart": "samples/kokoro_voices/af_heart.wav",
    "af_bella": "samples/kokoro_voices/af_bella.wav",
    "am_adam": "samples/kokoro_voices/am_adam.wav",
    "ef_dora": "samples/kokoro_voices/ef_dora.wav",
    "ff_siwis": "samples/kokoro_voices/ff_siwis.wav",
    "if_sara": "samples/kokoro_voices/if_sara.wav",
    "pf_dora": "samples/kokoro_voices/pf_dora.wav",
    "de_sarah": "samples/kokoro_voices/de_sarah.wav",
    "ru_nicole": "samples/kokoro_voices/ru_nicole.wav",
    "tr_river": "samples/kokoro_voices/tr_river.wav",
    "hi_puck": "samples/kokoro_voices/hi_puck.wav",
    "hf_alpha": "samples/kokoro_voices/hf_alpha.wav",
    "dave": "samples/neutts_air/dave.wav",
    "jo": "samples/neutts_air/jo.wav",
}

class ChatterboxWrapper:
    """
    Wrapper for Chatterbox TTS models (Turbo variant).
    
    NOTE: As of early 2026, the Turbo variant is primarily optimized for English.
    """
    
    def __init__(self, device: str | None = None, model_type: str = "chatterbox-turbo", multilingual: bool = False):
        self.device = device or self._auto_detect_device()
        self.multilingual = multilingual
        self.model_type = model_type
        
        if multilingual:
            logger.warning("Chatterbox-Turbo currently does not support a dedicated multilingual model. "
                           "Falling back to English-only Turbo or use standard Chatterbox for multilingual.")
            self.multilingual = False

        repo_id = "ResembleAI/chatterbox-turbo"
        
        logger.info(f"Initializing {model_type} on {self.device}")
        
        try:
            import torch
            _orig_load = torch.load
            def _patched_load(f, *args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = 'cpu'
                return _orig_load(f, *args, **kwargs)

            from chatterbox.models.t3 import llama_configs
            TURBO_CONFIG = {
              "activation_function": "gelu_new",
              "architectures": ["GPT2LMHeadModel"],
              "attn_pdrop": 0.1,
              "bos_token_id": 50256,
              "embd_pdrop": 0.1,
              "eos_token_id": 50256,
              "initializer_range": 0.02,
              "layer_norm_epsilon": 1e-05,
              "model_type": "gpt2",
              "n_ctx": 8196,
              "n_embd": 1024,
              "hidden_size": 1024,
              "n_head": 16,
              "n_layer": 24,
              "n_positions": 8196,
              "vocab_size": 50276,
            }
            llama_configs.LLAMA_CONFIGS["Turbo"] = TURBO_CONFIG

            ckpt_dir = Path(snapshot_download(repo_id=repo_id))

            from chatterbox.tts import ChatterboxTTS, Conditionals
            from chatterbox.models.t3.modules.t3_config import T3Config
            from chatterbox.models.t3 import T3
            from chatterbox.models.s3gen import S3Gen
            from chatterbox.models.voice_encoder import VoiceEncoder
            from chatterbox.models.tokenizers import EnTokenizer

            import chatterbox.models.tokenizers.tokenizer as cb_tok_mod
            cb_tok_mod.SOT = "<|endoftext|>"
            cb_tok_mod.EOT = "<|endoftext|>"

            tok_path = ckpt_dir / "tokenizer.json"

            # Initialize components manually with Turbo config
            hp = T3Config(text_tokens_dict_size=50276)
            hp.speech_tokens_dict_size = 6563
            hp.llama_config_name = "Turbo"
            hp.input_pos_emb = None
            hp.use_perceiver_resampler = False
            hp.emotion_adv = False
            
            hp.start_text_token = 50256
            hp.stop_text_token = 50256
            hp.start_speech_token = 6561
            hp.stop_speech_token = 6562

            t3 = T3(hp)
            t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
            if "model" in t3_state.keys():
                t3_state = t3_state["model"][0]
            t3.load_state_dict(t3_state)
            t3.to(self.device).eval()

            ve = VoiceEncoder()
            ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
            ve.to(self.device).eval()

            s3gen = S3Gen()
            s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
            s3gen.to(self.device).eval()

            tokenizer = EnTokenizer(str(tok_path))

            conds = None
            if (builtin_voice := ckpt_dir / "conds.pt").exists():
                map_loc = 'cpu' if self.device in ['cpu', 'mps'] else None
                conds = Conditionals.load(builtin_voice, map_location=map_loc).to(self.device)

            self.model = ChatterboxTTS(t3, s3gen, ve, tokenizer, self.device, conds=conds)
                
            logger.info(f"✓ {model_type} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise
    
    def _auto_detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def generate(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language_id: str | None = None
    ) -> torch.Tensor:
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        
        # Use inference_turbo for speed
        text_tokens = self.model.tokenizer.text_to_tokens(text).to(self.device)
        
        speech_tokens = self.model.t3.inference_turbo(
            self.model.conds.t3, 
            text_tokens,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        from chatterbox.models.s3tokenizer import drop_invalid_tokens
        speech_tokens = drop_invalid_tokens(speech_tokens[0])
        speech_tokens = speech_tokens[speech_tokens < 6561].to(self.device)
        
        wav, _ = self.model.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.model.conds.gen)
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = self.model.watermarker.apply_watermark(wav, sample_rate=self.model.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
    
    @property
    def sr(self) -> int:
        return self.model.sr

def _synthesize_with_pytorch(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    multilingual: bool = False,
    model_type: str = "chatterbox-turbo",
    device: str | None = None
):
    wrapper = ChatterboxWrapper(device=device, model_type=model_type, multilingual=multilingual)
    wav = wrapper.generate(
        text,
        audio_prompt_path=source_wav,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        language_id=language
    )
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav
    if wav_np.ndim > 1:
        wav_np = wav_np.squeeze()
    sf.write(output_wav, wav_np, wrapper.sr)
    logger.info(f"✓ Saved audio to {output_wav}")

def _synthesize_with_mlx(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    model_type: str = "chatterbox-turbo",
    model_id: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    stream: bool = False
):
    try:
        from mlx_audio.tts.generate import generate_audio
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import ChatterboxTurboTTS as ChatterboxTurboModel
    except ImportError:
        try:
            from mlx_audio.tts.generate import generate_audio
            ChatterboxTurboModel = None
        except ImportError:
            raise ImportError(
                "MLX backend requires 'mlx-audio>=0.2.10' package. Install with:\n"
                "  uv pip install -U mlx-audio"
            )

    if ChatterboxTurboModel is not None:
        original_load_weights = ChatterboxTurboModel.load_weights
        def patched_load_weights(self, weights: list, strict: bool = False):
            original_load_weights(self, weights, strict=False)
        ChatterboxTurboModel.load_weights = patched_load_weights
    
    logger.info(f"Generating speech with MLX backend ({model_type})...")
    
    ref_audio = source_wav
    if not ref_audio and voice and voice in VOICE_PRESETS:
        ref_audio = VOICE_PRESETS[voice]
    elif not ref_audio and voice and os.path.exists(str(voice)):
        ref_audio = voice
    if not ref_audio:
        ref_audio = VOICE_PRESETS.get(language, VOICE_PRESETS.get("en"))
    if not ref_audio:
        raise ValueError(f"Chatterbox (MLX) requires a reference audio file.")

    target_model = model_id or "mlx-community/Chatterbox-Turbo-4bit"
    lang_code = language if language else "en"
    
    output_dir = os.path.dirname(output_wav) or "."
    output_name = os.path.splitext(os.path.basename(output_wav))[0]
    file_prefix = os.path.join(output_dir, output_name)
    
    voice_map = {
        "en": "af_heart", "es": "ef_dora", "fr": "ff_siwis", "it": "if_sara",
        "pt": "pf_dora", "de": "af_sarah", "ru": "af_nicole", "tr": "af_river",
        "hi": "hf_alpha", "ja": "jf_alpha", "zh": "zf_xiaoxiao"
    }
    kokoro_voice = voice if voice else voice_map.get(language, "af_heart")
    
    kwargs = {
        "text": text,
        "model": target_model,
        "ref_audio": ref_audio,
        "ref_text": ".",
        "file_prefix": file_prefix,
        "lang_code": lang_code,
        "voice": kokoro_voice,
        "speed": speed,
        "stream": False,
        "play": stream,
        "verbose": True
    }
    
    generate_audio(**kwargs)
    
    expected_output = f"{file_prefix}_000.wav"
    if not os.path.exists(expected_output) and os.path.exists(f"{file_prefix}.wav"):
        expected_output = f"{file_prefix}.wav"
    
    if os.path.exists(expected_output):
        if os.path.abspath(expected_output) != os.path.abspath(output_wav):
            if os.path.exists(output_wav):
                os.remove(output_wav)
            os.rename(expected_output, output_wav)
    else:
        raise RuntimeError(f"MLX generation failed: {expected_output} not found")

def synthesize_with_chatterbox_turbo(
    text: str,
    output_wav: str,
    source_wav: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    language: str | None = None,
    multilingual: bool = False,
    use_mlx: bool = False,
    model_id: str | None = None,
    voice: str | None = None,
    speed: float = 1.0,
    stream: bool = False,
    device: str | None = None
):
    if use_mlx:
         _synthesize_with_mlx(
             text=text, 
             output_wav=output_wav, 
             source_wav=source_wav, 
             exaggeration=exaggeration, 
             cfg_weight=cfg_weight, 
             language=language, 
             model_id=model_id,
             voice=voice,
             speed=speed,
             stream=stream
         )
    else:
        _synthesize_with_pytorch(
            text=text, 
            output_wav=output_wav, 
            source_wav=source_wav, 
            exaggeration=exaggeration, 
            cfg_weight=cfg_weight, 
            language=language, 
            multilingual=multilingual,
            device=device
        )
