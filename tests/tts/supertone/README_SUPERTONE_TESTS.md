# Supertone TTS Tests

Comprehensive test suite for the Supertone TTS implementation.

## Running Tests

### Run all tests:
```bash
uv run pytest tests/test_supertone.py -v
```

### Run specific test classes:
```bash
# Test model initialization and synthesis
uv run pytest tests/test_supertone.py::TestSupertoneTTS -v

# Test text processing
uv run pytest tests/test_supertone.py::TestTextProcessing -v
```

### Run specific tests:
```bash
# Test basic synthesis
uv run pytest tests/test_supertone.py::TestSupertoneTTS::test_basic_synthesis -v

# Test voice styles
uv run pytest tests/test_supertone.py::TestSupertoneTTS::test_synthesis_with_female_voices -v
```

## Test Coverage

### Model Tests (`TestSupertoneTTS`)
- ✅ Model initialization
- ✅ Voice style listing
- ✅ Voice style loading
- ✅ Basic synthesis
- ✅ Female voices (F1, F2)
- ✅ Male voices (M1, M2)
- ✅ Different inference steps (2, 5, 8, 12)
- ✅ Different speeds (0.8, 1.0, 1.2, 1.5)
- ✅ Long text handling
- ✅ Special characters
- ✅ Invalid voice fallback
- ✅ Convenience function

### Text Processing Tests (`TestTextProcessing`)
- ✅ Text tokenization
- ✅ Text normalization
- ✅ Mask generation

### File Tests
- ✅ Model files existence check

## Prerequisites

Tests require:
1. **Models downloaded**: `git clone https://huggingface.co/Supertone/supertonic models/supertonic`
2. **pytest installed**: `uv pip install pytest`
3. **onnxruntime installed**: `uv pip install onnxruntime`

## Notes

- Tests are automatically skipped if models are not downloaded
- Each test uses temporary files for output (auto-cleanup)
- Tests use reduced inference steps (4) for faster execution
- Long text tests may take more time and can be excluded with `-k "not long_text"`
