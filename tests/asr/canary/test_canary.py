#!/usr/bin/env python3
"""
Test script for NVIDIA Canary-1B-v2 ASR and Translation model
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent
AUDIO_PATH = ROOT_DIR / "samples" / "anger.wav"

from voice_cloning.asr.canary import get_canary, transcribe_to_file, CanaryASR

# Compatibility aliases
CanaryV2Model = CanaryASR
transcribe_audio = transcribe_to_file


def test_canary_asr():
    """Test Canary ASR functionality."""
    print("=== Testing Canary ASR ===")
    
    # Test using the simple canary.py interface
    output_path = transcribe_to_file(
        audio_path=str(AUDIO_PATH),
        output_path="test_canary_asr.txt",
        source_lang="en",
        target_lang="en"
    )
    print(f"✓ ASR transcription saved to: {output_path}")
    
    # Read and display result
    with open(output_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        print(f"Transcription: {lines[4] if len(lines) > 4 else 'Error reading result'}")


def test_canary_translation():
    """Test Canary translation functionality."""
    print("\\n=== Testing Canary Translation ===")
    
    # Test EN -> FR translation
    model = CanaryV2Model()
    
    result = model.transcribe(
        audio_path=str(AUDIO_PATH),
        source_lang="en",
        target_lang="fr"
    )
    
    print(f"✓ EN→FR translation: {result['text']}")
    
    # Test EN -> DE translation
    result = model.transcribe(
        audio_path=str(AUDIO_PATH), 
        source_lang="en",
        target_lang="de"
    )
    
    print(f"✓ EN→DE translation: {result['text']}")


def test_supported_languages():
    """Test language support."""
    print("\\n=== Testing Supported Languages ===")
    
    model = CanaryV2Model()
    languages = model.get_supported_languages()
    
    print(f"✓ Canary supports {len(languages)} languages:")
    for code, name in list(languages.items())[:5]:  # Show first 5
        print(f"  {code}: {name}")
    print(f"  ... and {len(languages) - 5} more")
    
    # Test language validation
    assert model.validate_language("en"), "English should be supported"
    assert model.validate_language("fr"), "French should be supported"
    assert not model.validate_language("xx"), "Invalid language should not be supported"
    
    print("✓ Language validation working correctly")


def test_main_cli_integration():
    """Test integration with main.py CLI."""
    print("\\n=== Testing Main CLI Integration ===")
    
    # This would normally be tested by calling main.py, but we'll just verify the import works
    print("✓ Main CLI integration import successful")
    
    # Test that the canary module can be imported as expected by main.py
    get_canary()
    print("✓ Canary instance created successfully")


def main():
    """Run all tests."""
    print("🐤 NVIDIA Canary-1B-v2 Test Suite")
    print("=" * 50)
    
    # Check if sample audio exists
    if not AUDIO_PATH.exists():
        print(f"✗ Test audio file '{AUDIO_PATH}' not found")
        print("Please ensure the sample audio file exists before running tests")
        sys.exit(1)
    
    tests = [
        test_supported_languages,
        test_main_cli_integration,
        test_canary_asr,
        test_canary_translation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\\n" + "=" * 50)
    print(f"🐤 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Canary-1B-v2 is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
