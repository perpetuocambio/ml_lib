"""Test NSFW LoRA selection improvement."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from ml_lib.diffusion.services.intelligent_builder import IntelligentPipelineBuilder

# Use the first prompt from sanitized list (explicit fellatio/cum content)
TEST_PROMPT = "score_9, score_8_up, score_7_up, score_6_up, ((excessive cum, huge facial, cum on nose, thick ropes of cum)), 1 girl, granny, freckles, sexy, blushing, slim, kneeling, big facial, big cumshot, cum drip, small ass, smiling, soles, slight open mouth, head tilted up, portrait, bedroom, detailed drawing, partially illuminated, blurry background, face focus, from above, looking up at viewer, s1_dram, watercolor \\(medium\\), traditional media, (((cock on face, veiny cock))), ((wrinkled face, wrinkles buttoms, wrinkles tits, puffy nipples, creampie anal, creampie pussy, double penetration, 3 male offside camara, only cock visible. Mother Gothel))"

NEGATIVE = "worst quality, low quality, blurry, distorted, bad anatomy, text, watermark"

print("=" * 80)
print("TESTING NSFW LORA SELECTION")
print("=" * 80)
print()
print("Test: Prompt with explicit cum/facial content")
print("Expected: Should select cum/facial specific LoRAs")
print()

try:
    print("1. Initializing builder...")
    builder = IntelligentPipelineBuilder.from_comfyui_auto(
        enable_ollama=True,
        ollama_model="dolphin3",
        device="cuda",
    )
    print("✅ Builder initialized")
    print()

    print("2. Generating (watch LoRA selection)...")
    image = builder.generate(
        prompt=TEST_PROMPT,
        negative_prompt=NEGATIVE,
        quality="high",
        width=1024,
        height=1024,
        seed=123,
        steps=35,
        cfg_scale=7.5,
    )

    print("✅ Generation completed!")
    print()

    # Save
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nsfw_lora_test.png"

    if isinstance(image, list):
        image[0].save(output_path)
    else:
        image.save(output_path)

    print(f"✅ Image saved to: {output_path}")
    print()
    print("=" * 80)
    print("SUCCESS - Check image quality and LoRA relevance")
    print("=" * 80)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
