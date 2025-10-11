"""
Complete Generation Demo - All Features

Demonstrates:
- Auto model selection with Ollama analysis
- Style hints
- Quality levels
- Memory optimization
- Metadata embedding
- LoRA selection

Requirements:
- ComfyUI with models
- Ollama (optional, for intelligent selection)
"""

from pathlib import Path
from ml_lib.diffusion.services import (
    IntelligentPipelineBuilder,
    ImageNamingConfig,
)

# Create output directory
output_dir = Path("output/demo")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Intelligent Image Generation - Complete Demo")
print("=" * 60)

# Initialize builder with Ollama intelligence
print("\n[1] Initializing pipeline builder...")
builder = IntelligentPipelineBuilder.from_comfyui_auto(
    enable_ollama=True,  # Enable intelligent analysis
)

# Get stats
stats = builder.get_stats()
print(f"✓ Device: {stats['device']}")
print(f"✓ Ollama enabled: {stats['ollama_enabled']}")
print(f"✓ Available GPU memory: {stats['resources']['gpu_memory_gb']:.1f}GB")
print(f"✓ Models indexed: {stats['orchestrator']['total_models']}")

# Example 1: Simple generation (AUTO mode)
print("\n[2] Example 1: Simple AUTO generation")
print("-" * 60)
prompt1 = "a beautiful anime girl with pink hair in a magical forest"

image1 = builder.generate(
    prompt=prompt1,
    quality="balanced",  # fast | balanced | high | ultra
    width=1024,
    height=1024,
    seed=42,  # For reproducibility
)

if image1:
    image1.save(output_dir / "example1_auto.png")
    print(f"✓ Saved: example1_auto.png")
else:
    print("✗ Generation failed")

# Example 2: Style-guided generation
print("\n[3] Example 2: Style-guided generation")
print("-" * 60)
prompt2 = "photorealistic portrait of a woman, professional photography"

image2 = builder.generate(
    prompt=prompt2,
    style="realistic",  # Hint for model selection
    quality="high",  # Higher quality
    width=768,
    height=1024,
    seed=123,
)

if image2:
    image2.save(output_dir / "example2_realistic.png")
    print(f"✓ Saved: example2_realistic.png")
else:
    print("✗ Generation failed")

# Example 3: Batch generation
print("\n[4] Example 3: Batch generation (4 variations)")
print("-" * 60)
prompt3 = "cyberpunk city at night, neon lights, rain"

images3 = builder.generate(
    prompt=prompt3,
    style="artistic",
    quality="balanced",
    num_images=4,  # Generate 4 variations
)

if images3:
    for i, img in enumerate(images3):
        img.save(output_dir / f"example3_batch_{i+1}.png")
        print(f"✓ Saved: example3_batch_{i+1}.png")
else:
    print("✗ Generation failed")

# Example 4: Advanced parameters
print("\n[5] Example 4: Advanced parameter override")
print("-" * 60)
prompt4 = "fantasy landscape with castle and mountains"

image4 = builder.generate(
    prompt=prompt4,
    quality="ultra",
    steps=50,  # Override: more steps
    cfg_scale=9.0,  # Override: higher CFG
    sampler="DPM++ 2M SDE Karras",  # Override: specific sampler
)

if image4:
    image4.save(output_dir / "example4_advanced.png")
    print(f"✓ Saved: example4_advanced.png")
else:
    print("✗ Generation failed")

print("\n" + "=" * 60)
print("Demo complete! Check output/demo/ for generated images")
print("=" * 60)

# Show final stats
final_stats = builder.get_stats()
print(f"\nFinal GPU memory: {final_stats['resources']['gpu_memory_gb']:.1f}GB available")
