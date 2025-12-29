import os

def test_directory_structure_exists():
    base_dir = "tests"
    required_dirs = ["tts", "asr", "vad", "ui", "core", "data"]
    
    for d in required_dirs:
        path = os.path.join(base_dir, d)
        assert os.path.isdir(path), f"Directory {path} does not exist"
