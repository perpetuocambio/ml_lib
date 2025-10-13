#!/usr/bin/env python3
"""
Prueba REAL usando modelos LOCALES de ComfyUI
"""

print("🚀 Prueba REAL - Modelos Locales")
print("")

import torch
from diffusers import DiffusionPipeline
from PIL import Image
import time

print(f"✅ torch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print("")
print("=" * 70)
print("🎨 TEST 1: Generación con modelo local NSFW")
print("=" * 70)

# Usar modelo local NSFW
MODEL_PATH = "/src/ComfyUI/models/checkpoints/pornmaster_proSDXLV7.safetensors"

print(f"\n📥 Cargando modelo local: {MODEL_PATH}")
print("   (pornmaster_proSDXLV7 - optimizado para contenido adulto)")

try:
    pipe = DiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("✅ Pipeline en GPU")

    print("\n🎨 Generando imagen simple...")
    start = time.time()

    result = pipe(
        prompt="a beautiful sunset over mountains, vibrant colors, detailed, high quality",
        negative_prompt="low quality, blurry, bad",
        num_inference_steps=20,
        height=512,
        width=512,
    )

    elapsed = time.time() - start
    image = result.images[0]

    output_path = "/tmp/test_local_simple.png"
    image.save(output_path)

    print(f"✅ Imagen generada en {elapsed:.2f}s")
    print(f"📁 {output_path}")

    print("")
    print("=" * 70)
    print("🎨 TEST 2: Personaje Femenino")
    print("=" * 70)

    print("\n🎨 Generando personaje...")
    start = time.time()

    result = pipe(
        prompt="portrait of beautiful woman, long brown hair, green eyes, detailed face, professional photo, high quality",
        negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
        num_inference_steps=25,
        height=768,
        width=512,
    )

    elapsed = time.time() - start
    image = result.images[0]

    output_path = "/tmp/test_local_character.png"
    image.save(output_path)

    print(f"✅ Personaje generado en {elapsed:.2f}s")
    print(f"📁 {output_path}")

    print("")
    print("=" * 70)
    print("🎨 TEST 3: Contenido Adulto Artístico")
    print("=" * 70)

    print("\n🎨 Generando contenido adulto...")
    start = time.time()

    result = pipe(
        prompt="artistic nude portrait, beautiful woman, professional photography, tasteful lighting, elegant pose, high quality, detailed",
        negative_prompt="low quality, blurry, deformed, bad anatomy, distorted",
        num_inference_steps=30,
        height=1024,
        width=768,
    )

    elapsed = time.time() - start
    image = result.images[0]

    output_path = "/tmp/test_local_nsfw.png"
    image.save(output_path)

    print(f"✅ NSFW generado en {elapsed:.2f}s")
    print(f"📁 {output_path}")

    print("")
    print("=" * 70)
    print("🎨 TEST 4: Personaje Sexy/Sensual")
    print("=" * 70)

    print("\n🎨 Generando personaje sexy...")
    start = time.time()

    result = pipe(
        prompt="sexy woman in lingerie, beautiful face, seductive pose, bedroom setting, soft lighting, detailed, high quality",
        negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
        num_inference_steps=30,
        height=1024,
        width=768,
    )

    elapsed = time.time() - start
    image = result.images[0]

    output_path = "/tmp/test_local_sexy.png"
    image.save(output_path)

    print(f"✅ Personaje sexy generado en {elapsed:.2f}s")
    print(f"📁 {output_path}")

    print("")
    print("=" * 70)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("=" * 70)
    print("")
    print("📁 Imágenes generadas:")
    print("   1. /tmp/test_local_simple.png       - Paisaje simple")
    print("   2. /tmp/test_local_character.png    - Personaje femenino")
    print("   3. /tmp/test_local_nsfw.png         - Contenido adulto artístico")
    print("   4. /tmp/test_local_sexy.png         - Personaje sexy/sensual")
    print("")
    print("🎉 Sistema de generación funcionando perfectamente!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
