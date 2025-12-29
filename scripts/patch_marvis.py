import sys
import os

def patch_sesame():
    """
    Complete patch for mlx_audio sesame.py rope_scaling/rope_parameters mismatch.
    
    ROOT CAUSE:
    - HuggingFace config uses 'rope_scaling'
    - MLX's load_config() renames it to 'rope_parameters'
    - Code tries to access 'rope_scaling' → KeyError or None
    
    FIX:
    - Handle both 'rope_parameters' and 'rope_scaling' in two places:
      1. Top-level rope_scaling (line ~155)
      2. depth_decoder_config rope_scaling (line ~160)
    """
    import site
    
    site_packages = site.getsitepackages()[0]
    sesame_path = os.path.join(site_packages, 'mlx_audio', 'tts', 'models', 'sesame', 'sesame.py')
    
    if not os.path.exists(sesame_path):
        print(f"❌ Could not find sesame.py at {sesame_path}")
        return False
    
    with open(sesame_path) as f:
        content = f.read()
    
    if '# PATCHED: Complete rope_scaling/rope_parameters fix' in content:
        print("✓ File is already fully patched!")
        return True
    
    # Fix 1: Top-level rope_scaling pop
    buggy_pop = '''        depth_cfg = kwargs.pop("depth_decoder_config", None)
        rope_cfg = kwargs.pop("rope_scaling", None)'''
    
    fixed_pop = '''        depth_cfg = kwargs.pop("depth_decoder_config", None)
        # PATCHED: Complete rope_scaling/rope_parameters fix
        # MLX renames rope_scaling → rope_parameters, so check both
        rope_cfg = kwargs.pop("rope_parameters", kwargs.pop("rope_scaling", None))'''
    
    # Fix 2: depth_decoder_config rope_scaling access  
    buggy_depth = '''                    "rope_scaling": depth_cfg["rope_scaling"],'''
    fixed_depth = '''                    "rope_scaling": depth_cfg.get("rope_parameters", depth_cfg.get("rope_scaling")),'''
    
    changes = 0
    if buggy_pop in content:
        content = content.replace(buggy_pop, fixed_pop)
        changes += 1
        print("✓ Patched top-level rope_scaling")
    
    if buggy_depth in content:
        content = content.replace(buggy_depth, fixed_depth)
        changes += 1
        print("✓ Patched depth_decoder_config rope_scaling")
    
    if changes == 0:
        print("❌ Could not find code to patch. May already be partially patched.")
        return False
    
    try:
        with open(sesame_path, 'w') as f:
            f.write(content)
        print(f"\n✅ Successfully applied {changes} patches to {sesame_path}")
        print("\n📝 What was fixed:")
        print("  • Top-level: kwargs.pop('rope_parameters', kwargs.pop('rope_scaling', None))")
        print("  • Depth decoder: depth_cfg.get('rope_parameters', depth_cfg.get('rope_scaling'))")
        print("\n✓ Marvis should now work correctly!")
        return True
    except Exception as e:
        print(f"❌ Failed to write patch: {e}")
        return False

if __name__ == "__main__":
    success = patch_sesame()
    sys.exit(0 if success else 1)
