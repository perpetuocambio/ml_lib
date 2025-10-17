"""Test prompt compaction with real CivitAI prompts."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from ml_lib.diffusion.domain.services.intelligent_builder import IntelligentPipelineBuilder

# Get first prompt from sanitized file
prompt_file = Path(__file__).parent.parent / "data" / "prompt_sanitized.txt"
with open(prompt_file) as f:
    content = f.read()
    prompts = [p.strip() for p in content.split("-----------") if p.strip()]

# Use a simpler prompt first
TEST_PROMPT = prompts[1]  # Second prompt (30yo goth girl)

NEGATIVE = "worst quality, low quality, blurry, distorted, bad anatomy"

print("=" * 80)
print("TESTING PROMPT COMPACTION")
print("=" * 80)
print()
print(f"Original prompt length: {len(TEST_PROMPT)} chars")
print(f"Prompt: {TEST_PROMPT[:200]}...")
print()

try:
    # Create builder
    print("1. Initializing builder...")
    builder = IntelligentPipelineBuilder.from_comfyui_auto(
        enable_ollama=True,
        ollama_model="dolphin3",
        device="cuda",
    )
    print("✅ Builder initialized")
    print()

    # Generate
    print("2. Starting generation (this will compact the prompt automatically)...")
    image = builder.generate(
        prompt=TEST_PROMPT,
        negative_prompt=NEGATIVE,
        quality="high",
        width=1024,
        height=1024,
        seed=42,
        steps=30,
        cfg_scale=7.5,
    )

    print("✅ Generation completed!")
    print()

    # Save
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "compaction_test.png"

    if isinstance(image, list):
        image[0].save(output_path)
    else:
        image.save(output_path)

    print(f"✅ Image saved to: {output_path}")
    print()
    print("=" * 80)
    print("SUCCESS - Check the generated image")
    print("=" * 80)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
