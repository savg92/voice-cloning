import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging

logger = logging.getLogger(__name__)

class WhisperASR:
    def __init__(self, model_id: str = None, device: str = None, use_mlx: bool = False):
        self.use_mlx = use_mlx
        
        if device:
            self.device = device
        else:
            if self.use_mlx:
                self.device = "mps"
            else:
                self.device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Default model selection logic
        if not model_id:
            if self.use_mlx:
                self.model_id = "mlx-community/whisper-large-v3-turbo"
            else:
                default_model = (
                    "openai/whisper-tiny" if not torch.cuda.is_available() else "openai/whisper-large-v3"
                )
                self.model_id = os.environ.get("WHISPER_MODEL") or default_model
        else:
            self.model_id = model_id

        # Automatic mapping for MLX models
        # ONLY do this if use_mlx is True, otherwise Transformers will fail to load MLX weights
        if self.use_mlx and self.model_id and self.model_id.startswith("openai/"):
            # Map openai/whisper-large-v3-turbo -> mlx-community/whisper-large-v3-turbo
            mapped_id = self.model_id.replace("openai/", "mlx-community/")
            logger.info(f"Mapping {self.model_id} to {mapped_id} for MLX usage.")
            self.model_id = mapped_id
        elif not self.use_mlx and self.model_id and self.model_id.startswith("mlx-community/"):
             # Conversely, if someone passed mlx-community/ but is using PyTorch, map back
             mapped_id = self.model_id.replace("mlx-community/", "openai/")
             logger.info(f"Mapping {self.model_id} back to {mapped_id} for PyTorch usage.")
             self.model_id = mapped_id
            
        self.pipe = None

    def load_model(self):
        if self.use_mlx:
            # MLX Whisper loads model during transcribe or has its own management
            return

        if self.pipe is not None:
            return

        logger.info(f"Loading Whisper model {self.model_id} on {self.device} (dtype={self.torch_dtype}) ...")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=30,
            batch_size=8 if torch.cuda.is_available() else 1,
        )

    def transcribe(self, audio_path: str, lang: str = None, task: str = "transcribe", timestamps: bool = True) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.use_mlx:
            import mlx_whisper
            import mlx_whisper.whisper
            
            # Monkey-patch ModelDimensions to ignore extra kwargs (like _name_or_path)
            # which causes "got an unexpected keyword argument" error in some models.
            if not hasattr(mlx_whisper.whisper.ModelDimensions, "_is_patched"):
                original_init = mlx_whisper.whisper.ModelDimensions.__init__
                def patched_init(self, *args, **kwargs):
                    # Filter kwargs to only include those expected by the dataclass
                    valid_keys = self.__annotations__.keys()
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
                    original_init(self, *args, **filtered_kwargs)
                mlx_whisper.whisper.ModelDimensions.__init__ = patched_init
                mlx_whisper.whisper.ModelDimensions._is_patched = True

            logger.info(f"Transcribing {audio_path} with MLX Whisper ({self.model_id})...")
            
            # mlx_whisper.transcribe returns a dict with 'text' and 'segments'
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=self.model_id,
                language=lang,
                task=task,
                verbose=False
            )
            return result["text"]
        else:
            self.load_model()
            
            generation_kwargs = {
                "num_beams": 1,
                "max_new_tokens": 225,
            }
            
            if lang:
                generation_kwargs["language"] = lang
                generation_kwargs["task"] = task

            logger.info(f"Transcribing {audio_path}...")
            with torch.inference_mode():
                result = self.pipe(audio_path, return_timestamps=timestamps, generate_kwargs=generation_kwargs)
                
            return result["text"]

def transcribe_to_file(audio_path: str, output_path: str, language: str = None, task: str = "transcribe", timestamps: bool = False, use_mlx: bool = False, model_id: str = None):
    model = WhisperASR(model_id=model_id, use_mlx=use_mlx)
    
    result = None
    if use_mlx:
        import mlx_whisper
        # Use provided model ID or default based on MLX flag (handled in WhisperASR)
        mid = model.model_id
        logger.info(f"Transcribing {audio_path} with MLX Whisper ({mid}) for file output...")
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=mid,
            language=language,
            task=task,
            verbose=False
        )
        text = result["text"]
    else:
        model.load_model()
        result = model.pipe(
            audio_path, 
            return_timestamps=timestamps, 
            generate_kwargs={"language": language, "task": task} if language else {"task": task}
        )
        text = result["text"]
    
    # If timestamps requested, append them to text
    if timestamps and result:
        text += "\n\n--- Timestamps ---\n"
        if use_mlx and "segments" in result:
            for seg in result["segments"]:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                seg_text = seg.get("text", "").strip()
                text += f"[{start:.2f}s -> {end:.2f}s] {seg_text}\n"
        elif not use_mlx and "chunks" in result:
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp")
                if ts:
                    start, end = ts
                    chunk_text = chunk["text"]
                    text += f"[{start:.2f}s -> {end:.2f}s] {chunk_text}\n"

    with open(output_path, "w") as f:
        f.write(text)
        
    return text
