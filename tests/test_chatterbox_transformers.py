from transformers import AutoModel

print("Attempting to load ResembleAI/chatterbox via transformers...")
try:
    # Try loading as generic model or specific TTS model
    # Note: AutoModelForTextToSpeech might not support it if it's a custom architecture not in main transformers yet.
    # But let's try.
    model_id = "ResembleAI/chatterbox"
    
    # Try trust_remote_code=True as it's likely a custom model
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Successfully loaded model via AutoModel")
    
    # Check if generate accepts lang/exaggeration
    import inspect
    sig = inspect.signature(model.generate)
    print(f"Generate signature: {sig}")
    
except Exception as e:
    print(f"✗ Failed to load via AutoModelForTextToSpeech: {e}")

    try:
        # Try AutoModel
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        print("✓ Successfully loaded model via AutoModel")
    except Exception as e2:
        print(f"✗ Failed to load via AutoModel: {e2}")
