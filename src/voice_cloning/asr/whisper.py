import os
import sys
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
            torch_dtype=self.torch_dtype,
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
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=30,
            batch_size=8 if torch.cuda.is_available() else 1,
        )

    def transcribe(self, audio_path: str, lang: str = None, task: str = "transcribe", timestamps: bool = True) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.use_mlx:
            import mlx_whisper
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
    
    if use_mlx:
        text = model.transcribe(audio_path, lang=language, task=task, timestamps=timestamps)
        # MLX whisper transcribe returns text directly in our wrapper
    else:
        model.load_model()
        result = model.pipe(
            audio_path, 
            return_timestamps=timestamps, 
            generate_kwargs={"language": language, "task": task} if language else {"task": task}
        )
        text = result["text"]
    
    # If timestamps requested, append them to text or save separately?
    # For consistency with other tools, let's append them if it's a txt file, or just save text.
    # The user asked for "all features", so let's try to be helpful.
    
    if timestamps and not use_mlx and "chunks" in result:
        text += "\n\n--- Timestamps ---\n"
        for chunk in result["chunks"]:
            start, end = chunk["timestamp"]
            chunk_text = chunk["text"]
            text += f"[{start:.2f}s -> {end:.2f}s] {chunk_text}\n"

    with open(output_path, "w") as f:
        f.write(text)
        
    return text
