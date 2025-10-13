#!/usr/bin/env python3
"""
Prueba REAL de generación de imágenes.
"""

print("🚀 Iniciando prueba REAL de generación...")
print("")

# Test 1: Verificar imports
print("📦 Verificando imports...")
try:
    import torch
    print(f"✅ torch: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
except ImportError as e:
    print(f"❌ torch no disponible: {e}")
    exit(1)

try:
    from diffusers import DiffusionPipeline
    print("✅ diffusers instalado")
except ImportError as e:
    print(f"❌ diffusers no disponible: {e}")
    exit(1)

try:
    from PIL import Image
    print("✅ PIL instalado")
except ImportError as e:
    print(f"❌ PIL no disponible: {e}")
    exit(1)

print("")
print("=" * 60)
print("🎨 TEST 1: Generación Simple")
print("=" * 60)

# Crear pipeline simple
print("\n📥 Cargando modelo (esto puede tomar unos minutos la primera vez)...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("✅ Pipeline en GPU")
else:
    print("⚠️  Pipeline en CPU (será LENTO)")

print("\n🎨 Generando imagen simple...")
import time
start = time.time()

result = pipe(
    prompt="a beautiful sunset over mountains, vibrant colors, detailed",
    negative_prompt="low quality, blurry, bad",
    num_inference_steps=20,
    height=512,
    width=512,
)

elapsed = time.time() - start
image = result.images[0]

output_path = "/tmp/test_simple_generation.png"
image.save(output_path)

print(f"✅ Imagen generada en {elapsed:.2f}s")
print(f"📁 Guardada en: {output_path}")
print(f"   Resolución: {image.size}")

print("")
print("=" * 60)
print("🎨 TEST 2: Generación de Personaje")
print("=" * 60)

print("\n🎨 Generando personaje...")
start = time.time()

result = pipe(
    prompt="portrait of a beautiful woman with long brown hair, green eyes, detailed face, high quality",
    negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
    num_inference_steps=25,
    height=768,
    width=512,
)

elapsed = time.time() - start
image = result.images[0]

output_path = "/tmp/test_character.png"
image.save(output_path)

print(f"✅ Personaje generado en {elapsed:.2f}s")
print(f"📁 Guardada en: {output_path}")

print("")
print("=" * 60)
print("🎨 TEST 3: Contenido Adulto (NSFW)")
print("=" * 60)

print("\n🎨 Generando contenido artístico adulto...")
start = time.time()

result = pipe(
    prompt="artistic nude portrait, beautiful woman, professional photography, tasteful lighting, elegant",
    negative_prompt="low quality, blurry, deformed, bad anatomy",
    num_inference_steps=30,
    height=768,
    width=512,
)

elapsed = time.time() - start
image = result.images[0]

output_path = "/tmp/test_nsfw_artistic.png"
image.save(output_path)

print(f"✅ Contenido adulto generado en {elapsed:.2f}s")
print(f"📁 Guardada en: {output_path}")

print("")
print("=" * 60)
print("✅ TODAS LAS PRUEBAS COMPLETADAS")
print("=" * 60)
print("")
print("📁 Imágenes generadas:")
print("   - /tmp/test_simple_generation.png")
print("   - /tmp/test_character.png")
print("   - /tmp/test_nsfw_artistic.png")
print("")
print("🎉 ¡Generación de imágenes funcionando correctamente!")
