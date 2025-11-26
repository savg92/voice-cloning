import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GraniteASR:
    def __init__(self, model_name: str = "ibm-granite/granite-speech-3.3-8b"):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
            
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading Granite model: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, 
            device_map="auto", 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        )
        logger.info(f"Granite model loaded on {self.device} with {self.torch_dtype}")

    def transcribe(self, audio_path: str) -> str:
        self.load_model()
        
        # Load audio
        wav, sr = torchaudio.load(audio_path, normalize=True)
        
        # Resample if necessary (Granite expects 16kHz)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000
            
        # Ensure mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Create prompt
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Run inference
        model_inputs = self.processor(prompt, wav[0], sampling_rate=sr, device=self.device, return_tensors="pt").to(self.device)
        
        # Generate
        model_outputs = self.model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)
        
        # Decode
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        output_text = self.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        
        return output_text[0]

def transcribe_file(audio_path: str, output_path: str):
    model = GraniteASR()
    transcript = model.transcribe(audio_path)
    with open(output_path, "w") as f:
        f.write(transcript)
    return transcript