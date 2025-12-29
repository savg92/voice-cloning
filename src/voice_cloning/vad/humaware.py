import torch
import torchaudio
import logging

logger = logging.getLogger(__name__)

class HumAwareVAD:
    """
    Wrapper for CuriousMonkey7/HumAware-VAD model.
    This model is a fine-tuned Silero VAD designed to distinguish speech from humming.
    """
    
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Initializing HumAwareVAD on {self.device}")
        
        # Load model from Hugging Face
        # HumAware-VAD is based on Silero, so we load it similarly or via torch.hub if supported,
        # but the model card suggests it's a .jit or .onnx file usually.
        # However, the user provided a HF link. Let's check how to load it.
        # Usually Silero models are loaded via torch.hub.load('snakers4/silero-vad', 'silero_vad')
        # But this is a custom fine-tune.
        # We will try to load the model file directly from HF if possible, or use the provided usage pattern.
        # Since I can't browse live during execution easily without interrupting, I will assume standard torch load 
        # or use the 'silero_vad' interface with custom weights if applicable.
        
        # For now, I will implement a robust loader that tries to download the model.jit from the repo.
        
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from huggingface_hub import hf_hub_download
            
            model_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename="HumAwareVAD.jit")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info("✓ HumAware VAD model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HumAwareVAD: {e}")
            raise

    def _get_speech_timestamps(
        self,
        audio: torch.Tensor,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
    ) -> list[dict[str, int]]:
        """
        Standalone implementation of get_speech_timestamps.
        Returns speech segments as list of dicts with 'start' and 'end' in samples.
        """
        if sampling_rate != 16000:
            raise ValueError("Only 16000 Hz sampling rate is supported")

        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        audio_length_samples = len(audio)
        
        # Get speech probabilities for each window
        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[current_start_sample: current_start_sample + window_size_samples]
            if len(chunk) < window_size_samples:
                # Pad if necessary
                chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
            
            # Get probability from model
            with torch.no_grad():
                speech_prob = model(chunk.unsqueeze(0), sampling_rate).item()
            speech_probs.append(speech_prob)

        triggered = False
        speeches = []
        current_speech = {}
        
        # Find speech segments
        for i, speech_prob in enumerate(speech_probs):
            current_sample = i * window_size_samples
            
            if speech_prob >= threshold and not triggered:
                # Start of speech
                triggered = True
                current_speech = {'start': current_sample}
                
            elif speech_prob < threshold and triggered:
                # Potential end of speech
                current_speech['end'] = current_sample
                
                # Check if speech is long enough
                if (current_speech['end'] - current_speech['start']) >= min_speech_samples:
                    # Add padding
                    current_speech['start'] = max(0, current_speech['start'] - speech_pad_samples)
                    current_speech['end'] = min(audio_length_samples, current_speech['end'] + speech_pad_samples)
                    speeches.append(current_speech)
                
                triggered = False
                current_speech = {}

        # Handle case where speech extends to end of audio
        if triggered:
            current_speech['end'] = audio_length_samples
            if (current_speech['end'] - current_speech['start']) >= min_speech_samples:
                current_speech['start'] = max(0, current_speech['start'] - speech_pad_samples)
                speeches.append(current_speech)

        # Merge segments that are too close
        if len(speeches) > 0:
            merged_speeches = [speeches[0]]
            for speech in speeches[1:]:
                prev_speech = merged_speeches[-1]
                if speech['start'] - prev_speech['end'] < min_silence_samples:
                    # Merge
                    prev_speech['end'] = speech['end']
                else:
                    merged_speeches.append(speech)
            speeches = merged_speeches

        # Convert to integers
        for speech in speeches:
            speech['start'] = int(speech['start'])
            speech['end'] = int(speech['end'])

        return speeches

    def detect_speech(self, audio_path: str, threshold: float = 0.5, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100, speech_pad_ms: int = 30) -> list[dict]:
        """
        Detect speech segments in the audio file.
        
        Args:
            audio_path: Path to the audio file.
            threshold: Probability threshold for speech detection (0.0-1.0).
            min_speech_duration_ms: Minimum duration of speech chunks in ms.
            min_silence_duration_ms: Minimum duration of silence chunks in ms.
            speech_pad_ms: Padding to add to speech chunks in ms.
            
        Returns:
            List of dicts with 'start' and 'end' timestamps (in seconds).
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        # Use soundfile instead of torchaudio.load to avoid torchcodec dependency
        import soundfile as sf
        import numpy as np
        
        audio_np, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to torch tensor
        if len(audio_np.shape) > 1:
            # Stereo to mono
            audio_np = np.mean(audio_np, axis=1)
        
        wav = torch.from_numpy(audio_np)
        
        # Silero expects 16k sample rate
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
            sr = 16000
            
        wav = wav.to(self.device)
        
        # Get timestamps using standalone implementation
        timestamps = self._get_speech_timestamps(
            wav,
            self.model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            sampling_rate=sr
        )
        
        # Convert to seconds
        results = []
        for ts in timestamps:
            results.append({
                'start': ts['start'] / sr,
                'end': ts['end'] / sr
            })
            
        return results
