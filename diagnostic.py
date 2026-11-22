# diagnostic.py — Run this to check your model vs mapping alignment
import tensorflow as tf
import joblib
import numpy as np

print("=" * 50)
print("SILENTSPEAK MODEL DIAGNOSTIC")
print("=" * 50)

# 1. Check model output shape
try:
    model = tf.keras.models.load_model("models/akash_v2_final_model.keras")
    print(f"\n✓ Model loaded")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Expected classes: {model.output_shape[-1]}")
except Exception as e:
    print(f"\n✗ Failed to load model: {e}")

# 2. Check mapping file
try:
    mapping = joblib.load("models/akash_v2_final_class_mapping.pkl")
    print(f"\n✓ Mapping loaded")
    print(f"  Type: {type(mapping)}")
    print(f"  Contents: {mapping}")
    
    if isinstance(mapping, dict):
        print(f"  Entries: {len(mapping)}")
        print(f"  Max index: {max(mapping.keys()) if mapping else 'N/A'}")
        print(f"  Words: {list(mapping.values())}")
except Exception as e:
    print(f"\n✗ Failed to load mapping: {e}")

# 3. Check for mismatch
try:
    model_classes = model.output_shape[-1]
    map_classes = len(mapping) if isinstance(mapping, dict) else "N/A"
    print(f"\n--- COMPARISON ---")
    print(f"Model output classes: {model_classes}")
    print(f"Mapping entries:      {map_classes}")
    
    if isinstance(mapping, dict) and model_classes != len(mapping):
        print(f"\n⚠️ MISMATCH! Model expects {model_classes} classes but mapping has {len(mapping)}")
        print("This is why you get 'unknown' — indices go out of range.")
except:
    pass