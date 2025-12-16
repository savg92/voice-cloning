from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, Union
import numpy as np

class BenchmarkType:
    TTS = "TTS"
    ASR = "ASR"
    VAD = "VAD"

class ModelBenchmark(ABC):
    """Abstract base class for model benchmarks"""
    
    def __init__(self, model_name: str, benchmark_type: str, **kwargs):
        self.model_name = model_name
        self.type = benchmark_type
        self.kwargs = kwargs
        self.model = None

    @abstractmethod
    def load(self):
        """Load the model"""
        pass

    @abstractmethod
    def warmup(self, output_dir: str):
        """Run a warmup pass"""
        pass

    @abstractmethod
    def run_test(self, input_data: Any, output_path: str) -> Dict[str, Any]:
        """
        Run the benchmark test.
        
        Args:
            input_data: 
                - For TTS: text string
                - For ASR/VAD: path to audio file
            output_path: Path to write output (audio or text file)
            
        Returns:
            Dict containing relevant outputs, e.g.:
            - TTS: {'audio': np.ndarray, 'sr': int}
            - ASR: {'transcription': str}
            - VAD: {'segments': List[Dict]}
        """
        pass

    def cleanup(self):
        """Optional cleanup"""
        self.model = None
