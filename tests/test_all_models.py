#!/usr/bin/env python3
"""
Test script to demonstrate all voice cloning models.

This script tests each implemented model with sample text and reference audio.
It serves as both a demo and a validation tool for the voice cloning project.
"""

import os
import sys
from pathlib import Path

def test_model(model_name, text, reference=None, output_suffix="test"):
    """Test a specific model with given parameters."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} Model")
    print(f"{'='*60}")
    
    output_file = f"{model_name}_{output_suffix}_output.wav"
    
    # Build command
    cmd_parts = [
        "uv", "run", "python", "main.py",
        "--model", model_name,
        "--text", f'"{text}"',
        "--output", output_file
    ]
    
    if reference and os.path.exists(reference):
        cmd_parts.extend(["--reference", reference])
    
    if model_name == "openvoice2":
        cmd_parts.extend(["--language", "EN_NEWEST"])
    elif model_name == "kokoro":
        cmd_parts.extend(["--voice", "af_heart", "--speed", "1.0"])
    
    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}")
    print(f"Expected output: {output_file}")
    
    # Execute command
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ SUCCESS: Generated {output_file} ({file_size} bytes)")
            return True
        else:
            print(f"❌ FAILED: Output file {output_file} not created")
            return False
    else:
        print(f"❌ FAILED: Command exited with code {exit_code}")
        return False

def main():
    """Run all model tests."""
    print("Voice Cloning Models Test Suite")
    print("===============================")
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: Please run this script from the voice-cloning project root directory")
        sys.exit(1)
    
    # Test parameters
    test_text = "Hello, this is a comprehensive test of our voice cloning technology."
    reference_audio = "samples/anger.wav"
    
    # Check if reference audio exists
    if not os.path.exists(reference_audio):
        print(f"⚠️  Warning: Reference audio {reference_audio} not found.")
        print("   Voice cloning models will be skipped.")
        reference_audio = None
    
    # Test results
    results = {}
    
    # Test each model
    models_to_test = [
        ("kokoro", False),      # Kokoro doesn't need reference audio
        ("openvoice", True),    # OpenVoice v1 needs reference audio
        ("openvoice2", True),   # OpenVoice v2 needs reference audio
        ("chatterbox", True)    # Chatterbox needs reference audio
    ]
    
    for model, needs_reference in models_to_test:
        if needs_reference and not reference_audio:
            print(f"\n⏭️  Skipping {model} (no reference audio)")
            results[model] = "skipped"
            continue
        
        success = test_model(
            model, 
            test_text, 
            reference_audio if needs_reference else None
        )
        results[model] = "success" if success else "failed"
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for model, result in results.items():
        status_icon = {
            "success": "✅",
            "failed": "❌", 
            "skipped": "⏭️"
        }.get(result, "❓")
        
        print(f"{status_icon} {model.upper()}: {result}")
    
    # Overall result
    successful = sum(1 for r in results.values() if r == "success")
    total_attempted = sum(1 for r in results.values() if r != "skipped")
    
    print(f"\nOverall: {successful}/{total_attempted} models working successfully")
    
    if successful > 0:
        print("\n🎉 At least one model is working! Check the generated audio files.")
    
    # List generated files
    generated_files = [f for f in os.listdir(".") if f.endswith("_test_output.wav")]
    if generated_files:
        print(f"\nGenerated files:")
        for file in sorted(generated_files):
            size = os.path.getsize(file)
            print(f"  - {file} ({size} bytes)")

if __name__ == "__main__":
    main()
