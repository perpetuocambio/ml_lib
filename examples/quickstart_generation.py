"""
Minimal Quickstart Example - Intelligent Image Generation

This demonstrates the simplest possible usage:
- 5 lines of code
- Zero configuration
- Automatic model selection
- Automatic memory optimization

Requirements:
- ComfyUI installed with models (or models in standard paths)
- Optional: Ollama running for intelligent analysis

Example:
    python examples/quickstart_generation.py
"""

from ml_lib.diffusion.intelligent.pipeline.services import IntelligentPipelineBuilder

# Step 1: Create builder (auto-detects ComfyUI models)
builder = IntelligentPipelineBuilder.from_comfyui_auto(
    enable_ollama=False  # Set to True if you have Ollama running
)

# Step 2: Generate!
image = builder.generate("a beautiful sunset over mountains")

# Step 3: Save
image.save("output/quickstart_sunset.png")

print("✅ Image generated: output/quickstart_sunset.png")
