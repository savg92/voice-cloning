import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test MLX import
try:
    from mlx_lm import load
    print("✓ mlx_lm imported successfully")
    
    # Try to load the model
    print("Loading nhe-ai/maya1-mlx-4Bit...")
    model, tokenizer = load("nhe-ai/maya1-mlx-4Bit")
    print(f"✓ Model loaded! Type: {type(model)}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
