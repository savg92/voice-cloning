#!/usr/bin/env python3
"""
Test script for NVIDIA Canary-1B-v2 ASR and Translation model
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import importlib.util

from voice_cloning.asr.canary import get_canary, transcribe_to_file

# Import the canary-1b-v2 module using importlib due to hyphens in name
spec = importlib.util.spec_from_file_location(
    "canary_1b_v2", 
    Path(__file__).parent.parent / "src" / "voice_cloning" / "asr" / "canary_1b_v2.py"
)
canary_1b_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary_1b_v2)

CanaryV2Model = canary_1b_v2.CanaryV2Model
transcribe_audio = canary_1b_v2.transcribe_audio


def test_canary_asr():
    """Test Canary ASR functionality."""
    print("=== Testing Canary ASR ===")
    
    # Test using the simple canary.py interface
    try:
        output_path = transcribe_to_file(
            audio_path="sample_voices/anger.wav",
            output_path="test_canary_asr.txt",
            source_lang="en",
            target_lang="en"
        )
        print(f"✓ ASR transcription saved to: {output_path}")
        
        # Read and display result
        with open(output_path, 'r') as f:
            content = f.read()
            print(f"Transcription: {content.split('\\n')[4] if len(content.split('\\n')) > 4 else 'Error reading result'}")
            
    except Exception as e:
        print(f"✗ ASR test failed: {e}")
        return False
    
    return True


def test_canary_translation():
    """Test Canary translation functionality."""
    print("\\n=== Testing Canary Translation ===")
    
    try:
        # Test EN -> FR translation
        model = CanaryV2Model()
        
        result = model.transcribe_single(
            audio_path="sample_voices/anger.wav",
            source_lang="en",
            target_lang="fr"
        )
        
        print(f"✓ EN→FR translation: {result['text']}")
        
        # Test EN -> DE translation
        result = model.transcribe_single(
            audio_path="sample_voices/anger.wav", 
            source_lang="en",
            target_lang="de"
        )
        
        print(f"✓ EN→DE translation: {result['text']}")
        
    except Exception as e:
        print(f"✗ Translation test failed: {e}")
        return False
    
    return True


def test_supported_languages():
    """Test language support."""
    print("\\n=== Testing Supported Languages ===")
    
    try:
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
        
    except Exception as e:
        print(f"✗ Language test failed: {e}")
        return False
    
    return True


def test_main_cli_integration():
    """Test integration with main.py CLI."""
    print("\\n=== Testing Main CLI Integration ===")
    
    try:
        # This would normally be tested by calling main.py, but we'll just verify the import works
        from voice_cloning.canary import transcribe_to_file
        print("✓ Main CLI integration import successful")
        
        # Test that the canary module can be imported as expected by main.py
        canary = get_canary()
        print("✓ Canary instance created successfully")
        
    except Exception as e:
        print(f"✗ CLI integration test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🐤 NVIDIA Canary-1B-v2 Test Suite")
    print("=" * 50)
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    # Check if sample audio exists
    if not Path("sample_voices/anger.wav").exists():
        print("✗ Test audio file 'sample_voices/anger.wav' not found")
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
            if test():
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
