#!/usr/bin/env python3
"""
Prueba SIMPLE y RÁPIDA de generación con modelo ligero.
Usa un modelo pequeño para verificar que todo funciona.
"""

print("🚀 Test Simple de Generación")
print("=" * 60)

# Test imports
print("\n📦 Verificando imports...")
try:
    import torch
    print(f"✅ torch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Error: {e}")
    exit(1)

try:
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    print("✅ diffusers y PIL disponibles")
except ImportError as e:
    print(f"❌ Error: {e}")
    exit(1)

print("\n📥 Cargando modelo ligero (runwayml/stable-diffusion-v1-5)...")
print("   Nota: Este es un modelo más pequeño y estable")

# Usar modelo SD 1.5 que es más compatible
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,  # Desactivar safety checker para esta prueba
    requires_safety_checker=False
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("✅ Pipeline cargado en GPU")
else:
    print("⚠️  Pipeline en CPU (será lento)")

print("\n🎨 Generando imagen de prueba...")
print("   Prompt: 'a beautiful landscape with mountains and lake'")

import time
start = time.time()

try:
    result = pipe(
        prompt="a beautiful landscape with mountains and lake, sunset, photorealistic",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=20,  # Reducido para velocidad
        height=512,
        width=512,
        guidance_scale=7.5
    )

    elapsed = time.time() - start
    image = result.images[0]

    output_path = "/tmp/test_simple.png"
    image.save(output_path)

    print(f"\n✅ ¡Generación exitosa!")
    print(f"   Tiempo: {elapsed:.2f}s")
    print(f"   Archivo: {output_path}")
    print(f"   Tamaño: {image.size}")

    print("\n🎉 ¡Test completado exitosamente!")

except Exception as e:
    print(f"\n❌ Error durante generación: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ GENERACIÓN FUNCIONAL")
print("=" * 60)
