"""
Auto-Download Demo - Automatic Model Acquisition

Demonstrates the intelligent model download feature:
- Searches local models first
- If not found, searches HuggingFace Hub
- If still not found, searches CivitAI
- Automatically downloads and caches models
- Verifies SHA256 checksums

This enables "zero-installation" image generation - just run and the
system will download everything it needs automatically.

Requirements:
- Internet connection
- Disk space for models (~6-12GB per model)
- Optional: Ollama for intelligent selection

Example:
    python examples/auto_download_demo.py
"""

from pathlib import Path
from ml_lib.diffusion.services import IntelligentPipelineBuilder

# Create output directory
output_dir = Path("output/auto_download")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Auto-Download Demo - Zero Installation Image Generation")
print("=" * 60)

print("\n[INFO] This demo will:")
print("  1. Search for models locally (ComfyUI paths)")
print("  2. If not found, search HuggingFace Hub")
print("  3. If not found, search CivitAI")
print("  4. Automatically download the best match")
print("  5. Cache models in ~/.ml_lib/models/")
print("  6. Generate images using downloaded models")

print("\n" + "=" * 60)
print("Initializing Pipeline with Auto-Download ENABLED")
print("=" * 60)

# Initialize with auto-download enabled
# Note: This will create ~/.ml_lib/models/ cache directory
builder = IntelligentPipelineBuilder.from_comfyui_auto(
    enable_ollama=False,  # Set to True if you have Ollama
    enable_auto_download=True,  # ⚡ Enable auto-download!
)

print("\n✓ Pipeline initialized")
print("✓ ModelRegistry ready for auto-download")

# Example 1: Generate with auto-download
print("\n" + "=" * 60)
print("Example 1: Generate with Auto-Download")
print("=" * 60)

prompt = "a beautiful landscape with mountains and a lake, sunset, 4k"

print(f"\nPrompt: {prompt}")
print("\n[Generating...]")
print("If no models are found locally, the system will:")
print("  - Search HuggingFace for 'stable-diffusion-xl-base'")
print("  - Download automatically (may take 5-10 minutes)")
print("  - Cache for future use")
print("  - Generate your image!")

try:
    image = builder.generate(
        prompt=prompt,
        quality="balanced",
        width=1024,
        height=768,
    )

    if image:
        output_path = output_dir / "auto_download_example.png"
        image.save(output_path)
        print(f"\n✅ Success! Image saved to: {output_path}")
        print(f"   Model was automatically acquired and cached")
    else:
        print("\n✗ Generation failed")
        print("   Check logs above for details")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Check disk space (need ~6-12GB)")
    print("  3. Check ~/.ml_lib/models/ cache directory")
    print("  4. Try running with enable_ollama=False")

# Show cache stats
print("\n" + "=" * 60)
print("Cache Statistics")
print("=" * 60)

try:
    if builder.registry:
        stats = builder.registry.get_stats()
        print(f"\nTotal models in registry: {stats['total_models']}")
        print(f"Downloaded models: {stats['downloaded']}")
        print(f"Cache size: {stats['cache_size_gb']:.2f} GB")

        if stats["by_source"]:
            print("\nBy source:")
            for source, count in stats["by_source"].items():
                print(f"  - {source}: {count}")

        if stats["by_type"]:
            print("\nBy type:")
            for model_type, count in stats["by_type"].items():
                print(f"  - {model_type}: {count}")

        print(f"\nCache location: ~/.ml_lib/models/")
        print("Models are cached and reused across generations")

except Exception as e:
    print(f"Could not retrieve stats: {e}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print("\nKey Features Demonstrated:")
print("  ✓ Zero-installation model acquisition")
print("  ✓ Intelligent search (local → HF → CivitAI)")
print("  ✓ Automatic download with progress")
print("  ✓ SHA256 verification")
print("  ✓ Persistent caching")
print("\nNext time you run this, it will use cached models!")
