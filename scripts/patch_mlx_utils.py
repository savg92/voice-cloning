import sys
import os

def patch_mlx_audio_utils():
    """
    Patches mlx_audio/tts/utils.py to fix quantization predicate signature.
    
    ROOT CAUSE:
    - mlx_audio defines base_quant_requirements(p, m, config) (3 args)
    - mlx_lm calls the predicate with (p, m) (2 args)
    - This causes TypeError during quantization
    
    FIX:
    - Change base_quant_requirements to take (p, m) and capture config from closure
    """
    import site
    
    site_packages = site.getsitepackages()[0]
    utils_path = os.path.join(site_packages, 'mlx_audio', 'tts', 'utils.py')
    
    if not os.path.exists(utils_path):
        print(f"❌ Could not find utils.py at {utils_path}")
        return False
    
    with open(utils_path, 'r') as f:
        content = f.read()
    
    if '# PATCHED: Fixed quantization predicate signature' in content:
        print("✓ File is already patched!")
        return True
    
    # The buggy code
    buggy_code = '''    # Define base quantization requirements
    def base_quant_requirements(p, m, config):
        return (
            hasattr(m, "weight")
            and m.weight.shape[-1] % 64 == 0  # Skip layers not divisible by 64
            and hasattr(m, "to_quantized")
            and model_quant_predicate(p, m, config)
        )'''
    
    # The fixed code
    fixed_code = '''    # PATCHED: Fixed quantization predicate signature
    # mlx_lm calls predicate with (p, m), so we must capture config from closure
    def base_quant_requirements(p, m):
        return (
            hasattr(m, "weight")
            and m.weight.shape[-1] % 64 == 0  # Skip layers not divisible by 64
            and hasattr(m, "to_quantized")
            and model_quant_predicate(p, m, config)
        )'''
    
    if buggy_code not in content:
        print("❌ Could not find the exact buggy code pattern.")
        # Try to find it with slightly different formatting if needed
        return False
    
    content = content.replace(buggy_code, fixed_code)
    
    # Also need to fix the lambda for user-provided predicate
    buggy_lambda = '''        quant_predicate = lambda p, m, config: (
            base_quant_requirements(p, m, config) and original_predicate(p, m, config)
        )'''
    
    fixed_lambda = '''        # PATCHED: Fixed lambda signature
        quant_predicate = lambda p, m: (
            base_quant_requirements(p, m) and original_predicate(p, m, config)
        )'''
        
    if buggy_lambda in content:
        content = content.replace(buggy_lambda, fixed_lambda)
        print("✓ Patched user predicate lambda")
    
    try:
        with open(utils_path, 'w') as f:
            f.write(content)
        print(f"✅ Successfully patched {utils_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write patch: {e}")
        return False

if __name__ == "__main__":
    success = patch_mlx_audio_utils()
    sys.exit(0 if success else 1)
