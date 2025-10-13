"""
Test real NSFW generation with problematic prompt from user.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from ml_lib.diffusion.services.intelligent_builder import IntelligentPipelineBuilder

# User's problematic prompt
PROMPT = """an old female elephant in the forest, (masterpiece, best quality:1.2), amazing quality, very aesthetic, 32k, absurdres, extremely realistic, (anthro:1.5), an old elderly (female anthro elephant:1.9), (trunk nose suck own puffy nipple:1.6), blue eyes, big gaping pussy with creampie, spread pussy, UZ_full_bush, excessive pussy hair, orgasm face, eyes rolling, deep anal penetration, (anal penetration:1.6), bukkake, Large and open vulva filled with semen, The wrinkled and sagging breasts partially shown, (very wrinkled body:0.8), eyes rolling, cum cover all body, cum on anus, cum on ass, masterpiece, 4k, ray tracing, intricate details, highly-detailed, hyper-realistic, 8k RAW Editorial Photo, (face focus:0.8), BREAK 1male human, out of frame, (white cock:0.8)"""

NEGATIVE = """close-up, headshot, cropped face, exaggerated makeup, ugly, blur, cartoon, anime, doll, 3d, deformed, disfigured, nude, unrealistic, smooth skin, shiny skin, cgi, plastic, lowres, text, watermark, blurry, extra fingers, muscular, black cock, black man, man face, teenager, child, men faces, old men, black men, futa"""

print("="*80)
print("TESTING REAL NSFW GENERATION")
print("="*80)
print()

try:
    # Create builder
    print("1. Initializing IntelligentPipelineBuilder...")
    builder = IntelligentPipelineBuilder.from_comfyui_auto(
        enable_ollama=True,  # Activar Ollama para selección inteligente
        ollama_model="dolphin3",
        device="cuda",
    )
    print("✅ Builder initialized")
    print()

    # Get stats
    stats = builder.get_stats()
    print(f"GPU Memory: {stats['resources']['gpu_memory_gb']:.2f}GB")
    print(f"Available models: {stats['orchestrator']['total_models']}")
    print()

    # Generate
    print("2. Starting generation with prompt optimization...")
    print(f"Prompt (first 100 chars): {PROMPT[:100]}...")
    print()

    image = builder.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        quality="high",
        width=1024,
        height=1024,
        seed=42,
    )

    print("✅ Generation completed!")
    print()

    # Save
    output_path = Path("output_nsfw_test.png")
    if isinstance(image, list):
        image[0].save(output_path)
    else:
        image.save(output_path)

    print(f"✅ Image saved to: {output_path}")
    print()
    print("="*80)
    print("SUCCESS - Check the generated image for quality")
    print("="*80)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
