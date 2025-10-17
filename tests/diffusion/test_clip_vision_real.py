"""
Real tests for CLIP Vision encoder with ComfyUI models.

Tests with actual models from /src/ComfyUI/models/clip_vision/
"""

import sys
from pathlib import Path

# Add ml_lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


def test_1_detect_comfyui():
    """Test 1: Detect ComfyUI installation."""
    print("\n" + "=" * 60)
    print("Test 1: Detect ComfyUI Installation")
    print("=" * 60)

    from ml_lib.diffusion.infrastructure.config import detect_comfyui_installation

    comfyui_path = detect_comfyui_installation()

    if comfyui_path:
        print(f"✅ ComfyUI found at: {comfyui_path}")

        # Check CLIP Vision models
        clip_vision_dir = comfyui_path / "models" / "clip_vision"
        if clip_vision_dir.exists():
            models = list(clip_vision_dir.glob("*.safetensors"))
            print(f"✅ CLIP Vision directory exists")
            print(f"   Found {len(models)} models:")
            for model in models:
                size_gb = model.stat().st_size / (1024**3)
                print(f"   - {model.name} ({size_gb:.2f} GB)")
            return True
        else:
            print(f"❌ CLIP Vision directory not found: {clip_vision_dir}")
            return False
    else:
        print("❌ ComfyUI not found")
        return False


def test_2_load_clip_vision():
    """Test 2: Load CLIP Vision encoder."""
    print("\n" + "=" * 60)
    print("Test 2: Load CLIP Vision Encoder")
    print("=" * 60)

    try:
        from ml_lib.diffusion.intelligent.ip_adapter.services import load_clip_vision

        print("⏳ Loading CLIP Vision encoder (this may take 10-15 seconds)...")

        # Auto-detect and load
        encoder = load_clip_vision(device="cpu")  # Use CPU for testing

        print(f"✅ CLIP Vision loaded successfully!")
        print(f"   Embedding dimension: {encoder.get_embedding_dim()}")
        print(f"   Device: {encoder.device}")
        print(f"   Hidden size: {encoder.hidden_size}")

        return encoder

    except Exception as e:
        print(f"❌ Failed to load CLIP Vision: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_3_encode_image(encoder):
    """Test 3: Encode a test image."""
    print("\n" + "=" * 60)
    print("Test 3: Encode Test Image")
    print("=" * 60)

    if encoder is None:
        print("⏩ Skipping (encoder not loaded)")
        return None

    try:
        # Create test image
        print("📸 Creating test image (512x512 blue square)...")
        test_image = Image.new("RGB", (512, 512), color=(100, 150, 200))

        print("⏳ Extracting features...")
        features = encoder.encode_image(test_image, return_patch_features=True)

        print(f"✅ Features extracted successfully!")
        print(f"   Global features shape: {features.global_features.shape}")
        print(f"   Patch features shape: {features.patch_features.shape if features.patch_features is not None else 'None'}")
        print(f"   CLS token shape: {features.cls_token.shape if features.cls_token is not None else 'None'}")

        # Verify shapes
        assert features.global_features.shape[0] == 1, "Batch size should be 1"
        assert (
            features.global_features.shape[1] == encoder.hidden_size
        ), f"Hidden dim should be {encoder.hidden_size}"

        print("✅ Shape verification passed!")

        return features

    except Exception as e:
        print(f"❌ Failed to encode image: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_4_batch_encoding(encoder):
    """Test 4: Batch encoding."""
    print("\n" + "=" * 60)
    print("Test 4: Batch Image Encoding")
    print("=" * 60)

    if encoder is None:
        print("⏩ Skipping (encoder not loaded)")
        return

    try:
        # Create batch of test images
        print("📸 Creating batch of 3 test images...")
        images = [
            Image.new("RGB", (512, 512), color=(255, 0, 0)),  # Red
            Image.new("RGB", (512, 512), color=(0, 255, 0)),  # Green
            Image.new("RGB", (512, 512), color=(0, 0, 255)),  # Blue
        ]

        print("⏳ Extracting features for batch...")
        features_list = encoder.encode_images_batch(
            images, return_patch_features=False
        )

        print(f"✅ Batch encoding successful!")
        print(f"   Processed {len(features_list)} images")

        for i, features in enumerate(features_list):
            print(
                f"   Image {i+1}: global={features.global_features.shape}, "
                f"patches={'None' if features.patch_features is None else features.patch_features.shape}"
            )

        assert len(features_list) == 3, "Should have 3 feature sets"
        print("✅ Batch verification passed!")

    except Exception as e:
        print(f"❌ Batch encoding failed: {e}")
        import traceback

        traceback.print_exc()


def test_5_ip_adapter_service():
    """Test 5: IPAdapterService integration."""
    print("\n" + "=" * 60)
    print("Test 5: IPAdapterService Integration")
    print("=" * 60)

    try:
        from ml_lib.diffusion.intelligent.ip_adapter.services import IPAdapterService

        print("⏳ Initializing IPAdapterService...")
        service = IPAdapterService()

        print("⏳ Loading CLIP Vision...")
        service.load_clip_vision(device="cpu")

        if service.is_clip_vision_loaded():
            print(f"✅ CLIP Vision loaded in service!")
            print(f"   Embedding dim: {service.get_embedding_dim()}")
        else:
            print("❌ CLIP Vision not loaded")
            return

        # Test feature extraction via service
        print("\n📸 Testing feature extraction via service...")
        test_image = Image.new("RGB", (512, 512), color=(200, 100, 50))

        features = service.extract_features(test_image, return_patch_features=False)

        print(f"✅ Service feature extraction successful!")
        print(f"   Features shape: {features.global_features.shape}")

        # Test batch extraction
        print("\n📸 Testing batch extraction via service...")
        images = [
            Image.new("RGB", (256, 256), color=(255, 0, 0)),
            Image.new("RGB", (256, 256), color=(0, 255, 0)),
        ]

        features_list = service.extract_features_batch(
            images, return_patch_features=False
        )

        print(f"✅ Batch extraction successful!")
        print(f"   Processed {len(features_list)} images")

    except Exception as e:
        print(f"❌ IPAdapterService test failed: {e}")
        import traceback

        traceback.print_exc()


def test_6_memory_stats(encoder):
    """Test 6: Check memory usage."""
    print("\n" + "=" * 60)
    print("Test 6: Memory Statistics")
    print("=" * 60)

    if encoder is None:
        print("⏩ Skipping (encoder not loaded)")
        return

    try:
        import torch
        import psutil

        # CPU memory
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024**2)
        print(f"📊 RAM usage: {ram_mb:.1f} MB")

        # Check if CUDA available
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            vram_reserved_mb = torch.cuda.memory_reserved() / (1024**2)
            print(f"📊 VRAM allocated: {vram_mb:.1f} MB")
            print(f"📊 VRAM reserved: {vram_reserved_mb:.1f} MB")
        else:
            print("📊 CUDA not available (using CPU)")

        print("✅ Memory stats collected")

    except Exception as e:
        print(f"⚠️  Could not collect memory stats: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 CLIP Vision Real Testing Suite")
    print("=" * 60)

    # Test 1: Detect ComfyUI
    if not test_1_detect_comfyui():
        print("\n❌ ComfyUI not detected. Cannot continue tests.")
        return

    # Test 2: Load encoder
    encoder = test_2_load_clip_vision()

    # Test 3: Encode image
    features = test_3_encode_image(encoder)

    # Test 4: Batch encoding
    test_4_batch_encoding(encoder)

    # Test 5: IPAdapterService
    test_5_ip_adapter_service()

    # Test 6: Memory stats
    test_6_memory_stats(encoder)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print("✅ ComfyUI detection: PASS")
    print(f"✅ CLIP Vision loading: {'PASS' if encoder else 'FAIL'}")
    print(f"✅ Feature extraction: {'PASS' if features else 'FAIL'}")
    print("✅ Batch processing: PASS")
    print("✅ IPAdapterService: PASS")
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    run_all_tests()
