"""Quick test for CLIP Vision - Just detection and basic checks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Quick CLIP Vision Test")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from ml_lib.diffusion.infrastructure.config import detect_comfyui_installation, ModelPathConfig
    from ml_lib.diffusion.domain.value_objects_models import ModelType
    from ml_lib.diffusion.intelligent.ip_adapter.services import CLIPVisionEncoder
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Detect ComfyUI
print("\n2. Detecting ComfyUI...")
comfyui_path = detect_comfyui_installation()
if comfyui_path:
    print(f"✅ ComfyUI found: {comfyui_path}")

    clip_vision_dir = comfyui_path / "models" / "clip_vision"
    if clip_vision_dir.exists():
        models = list(clip_vision_dir.glob("*.safetensors"))
        print(f"✅ Found {len(models)} CLIP Vision models:")
        for model in models:
            size_gb = model.stat().st_size / (1024**3)
            print(f"   - {model.name} ({size_gb:.2f} GB)")
    else:
        print(f"❌ CLIP Vision dir not found: {clip_vision_dir}")
        sys.exit(1)
else:
    print("❌ ComfyUI not found")
    sys.exit(1)

# Test 3: ModelPathConfig
print("\n3. Testing ModelPathConfig...")
config = ModelPathConfig.from_root(comfyui_path)
clip_vision_paths = config.get_paths(ModelType.CLIP_VISION)
print(f"✅ Config loaded, CLIP Vision paths: {len(clip_vision_paths)}")

# Test 4: Check model file exists
print("\n4. Checking model file...")
clip_g = comfyui_path / "models" / "clip_vision" / "clip_vision_g.safetensors"
if clip_g.exists():
    print(f"✅ CLIP-G exists: {clip_g}")
    print(f"   Size: {clip_g.stat().st_size / (1024**3):.2f} GB")
else:
    print(f"⚠️  CLIP-G not found, trying CLIP-H...")
    clip_h = comfyui_path / "models" / "clip_vision" / "clip_vision_h.safetensors"
    if clip_h.exists():
        print(f"✅ CLIP-H exists: {clip_h}")
    else:
        print("❌ No CLIP Vision models found")
        sys.exit(1)

print("\n" + "=" * 60)
print("✅ All quick tests passed!")
print("=" * 60)
print("\nNote: Full model loading test skipped (takes ~1 min)")
print("To test loading, run: python tests/test_clip_vision_real.py")
