from pathlib import Path

# Constants
TEST_TEXT = "The quick brown fox jumps over the lazy dog. This is a benchmark test to measure synthesis speed."
OUTPUT_DIR = Path("outputs/benchmark")
BENCHMARK_FILE = "docs/BENCHMARK_RESULTS.md"

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
