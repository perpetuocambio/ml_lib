"""
Simple Generation Example - Zero Configuration.

This demonstrates the user-facing API: SIMPLE and EASY.

User provides: prompt + basic options
System handles: EVERYTHING else

No need to understand:
- Model architectures
- VAE compatibility
- LoRA selection
- Memory optimization
- Parameter tuning

Everything is automatic and intelligent.
"""

from ml_lib.diffusion.intelligent.pipeline import IntelligentPipelineBuilder

# ============================================================================
# EXAMPLE 1: Simplest usage - Auto-detect ComfyUI, just provide prompt
# ============================================================================

print("Example 1: Simplest generation")
print("=" * 60)

# Auto-detect ComfyUI installation and initialize
builder = IntelligentPipelineBuilder.from_comfyui_auto()

# Generate image - just provide prompt
image = builder.generate("a beautiful sunset over mountains")
image.save("sunset.png")

print("✅ Generated: sunset.png")
print()


# ============================================================================
# EXAMPLE 2: With style hint
# ============================================================================

print("Example 2: With style hint")
print("=" * 60)

# Style hints help select better models
image = builder.generate(
    prompt="a girl with pink hair",
    style="anime",  # Selects anime-optimized models
    quality="high",  # Uses more steps, better quality
)
image.save("anime_girl.png")

print("✅ Generated: anime_girl.png")
print()


# ============================================================================
# EXAMPLE 3: With Ollama intelligence (MOST INTELLIGENT)
# ============================================================================

print("Example 3: Ollama-powered generation")
print("=" * 60)

# Enable Ollama for semantic prompt analysis
builder_smart = IntelligentPipelineBuilder.from_comfyui_auto(enable_ollama=True)

# Ollama analyzes prompt and selects:
# - Best base model for "cyberpunk"
# - Relevant LoRAs (sci-fi, neon, urban)
# - Optimal generation parameters
image = builder_smart.generate("cyberpunk city at night with neon lights")
image.save("cyberpunk_city.png")

print("✅ Generated: cyberpunk_city.png")
print()


# ============================================================================
# EXAMPLE 4: Multiple images
# ============================================================================

print("Example 4: Batch generation")
print("=" * 60)

images = builder.generate(
    prompt="a fantasy castle on a hill",
    num_images=4,  # Generate 4 variations
    quality="balanced",
    seed=42,  # Reproducible
)

for i, img in enumerate(images):
    img.save(f"castle_{i}.png")

print(f"✅ Generated {len(images)} variations")
print()


# ============================================================================
# EXAMPLE 5: Custom paths (no ComfyUI)
# ============================================================================

print("Example 5: Custom model paths")
print("=" * 60)

# User can specify their own model directories
builder_custom = IntelligentPipelineBuilder.from_paths(
    model_paths={
        "checkpoint": ["/my/models/checkpoints"],
        "lora": ["/my/models/loras"],
        "vae": ["/my/models/vae"],
    },
    enable_ollama=True,
)

image = builder_custom.generate("a magical forest")
image.save("forest.png")

print("✅ Generated: forest.png")
print()


# ============================================================================
# EXAMPLE 6: Advanced - Override technical parameters
# ============================================================================

print("Example 6: Advanced overrides")
print("=" * 60)

# Advanced users can override technical parameters
image = builder.generate(
    prompt="a portrait of an old wizard",
    style="realistic",
    quality="ultra",
    # Technical overrides (optional)
    steps=50,  # More steps = better quality
    cfg_scale=9.0,  # Higher guidance
    sampler="DPM++ 2M Karras",  # Specific sampler
)
image.save("wizard.png")

print("✅ Generated: wizard.png")
print()


# ============================================================================
# Get statistics
# ============================================================================

print("System Statistics")
print("=" * 60)

stats = builder.get_stats()
print(f"Device: {stats['device']}")
print(f"Ollama enabled: {stats['ollama_enabled']}")
print(f"Available GPU memory: {stats['resources']['gpu_memory_gb']:.1f} GB")
print(f"Available RAM: {stats['resources']['ram_gb']:.1f} GB")

orchestrator_stats = stats['orchestrator']
print(f"\nIndexed models: {orchestrator_stats['total_models']}")
print(f"By type:")
for model_type, count in orchestrator_stats['by_type'].items():
    print(f"  - {model_type}: {count}")

print()
print("=" * 60)
print("✅ All examples completed!")
