
import pytest
import os
from pathlib import Path
from benchmarks.runner import BenchmarkResult, BenchmarkType

# We will implement this class
# from benchmarks.results_manager import BenchmarkResultsManager

class TestBenchmarkResultsManager:
    @pytest.fixture
    def mock_results_file(self, tmp_path):
        content = """# Benchmark Results
        
## Summary
Some summary here.

## Complete Benchmark Data

### All Models - Raw Data

| Model | Type | Latency (ms) | RTF | Memory (MB) | Audio Duration (s) | Speed Multiplier | Device |
|-------|------|--------------|-----|-------------|-------------------|------------------|--------|
| **Supertone** | TTS | 319 | 0.0459 | 293.1 | 6.95 | 21.8× | MPS |
| **KittenTTS Nano** | TTS | 1006 | 0.1302 | 67.5 | 7.73 | 7.7× | MPS |

## Text-to-Speech (TTS) Results

| Model | Latency | RTF | Memory | Speed Multiplier | Notes |
|-------|---------|-----|--------|------------------|-------|
| **Supertone** | 319ms | 0.046 | 293.1 MB | **21.8× real-time** | ONNX-based, ultra-fast |
"""
        file_path = tmp_path / "BENCHMARK_RESULTS.md"
        file_path.write_text(content)
        return file_path

    def test_update_existing_result(self, mock_results_file):
        from benchmarks.results_manager import BenchmarkResultsManager
        manager = BenchmarkResultsManager(mock_results_file)
        
        new_result = BenchmarkResult(
            model="Supertone",
            type="TTS",
            latency_ms=350.0,
            rtf=0.0500,
            memory_mb=300.0,
            notes="Device: mps"
        )
        
        # Additional fields might be needed for the complex table
        manager.update_result(new_result, audio_duration=7.0, device="MPS")
        
        updated_content = mock_results_file.read_text()
        assert "| **Supertone** | TTS | 350.00 | 0.0500 | 300.00 | 7.00 | 20.0× | MPS |" in updated_content
        assert "| **Supertone** | 350.00ms | 0.050 | 300.00 MB | **20.0× real-time** | Device: mps |" in updated_content
        # Ensure KittenTTS is still there
        assert "**KittenTTS Nano**" in updated_content

    def test_add_new_result(self, mock_results_file):
        from benchmarks.results_manager import BenchmarkResultsManager
        manager = BenchmarkResultsManager(mock_results_file)
        
        new_result = BenchmarkResult(
            model="Chatterbox Turbo",
            type="TTS",
            latency_ms=5000.0,
            rtf=0.5000,
            memory_mb=1200.0,
            notes="Device: mps"
        )
        
        manager.update_result(new_result, audio_duration=10.0, device="MPS")
        
        updated_content = mock_results_file.read_text()
        assert "| **Chatterbox Turbo** | TTS | 5000.00 | 0.5000 | 1200.00 | 10.00 | 2.0× | MPS |" in updated_content
        assert "| **Chatterbox Turbo** | 5000.00ms | 0.500 | 1200.00 MB | **2.0× real-time** | Device: mps |" in updated_content
