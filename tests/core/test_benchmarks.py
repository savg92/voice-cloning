import unittest
from benchmarks.tts.supertonic2 import Supertonic2Benchmark
from benchmarks.tts.supertone import SupertoneBenchmark
from benchmarks.base import BenchmarkType

class TestBenchmarks(unittest.TestCase):
    def test_supertonic2_benchmark_init(self):
        benchmark = Supertonic2Benchmark()
        self.assertEqual(benchmark.model_name, "Supertonic-2")
        self.assertEqual(benchmark.type, BenchmarkType.TTS)
        self.assertIsNone(benchmark.model_instance)

    def test_supertone_benchmark_init(self):
        benchmark = SupertoneBenchmark()
        self.assertEqual(benchmark.model_name, "Supertone")
        self.assertEqual(benchmark.type, BenchmarkType.TTS)

if __name__ == "__main__":
    unittest.main()
