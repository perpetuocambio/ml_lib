#!/usr/bin/env python3
"""
Prueba REAL usando modelos LOCALES - Compatible con safetensors
"""

print("🚀 Prueba REAL - Modelos Locales SDXL")
print("")

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import time
from pathlib import Path

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"✅ torch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print("")
print("=" * 70)
print("🎨 Cargando modelo NSFW local")
print("=" * 70)

# Usar modelo SDXL local optimizado para NSFW
MODEL_PATH = "/src/ComfyUI/models/checkpoints/pornmaster_proSDXLV7.safetensors"

print(f"\n📥 {MODEL_PATH}")
start_load = time.time()

pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

load_time = time.time() - start_load
print(f"✅ Modelo cargado en {load_time:.1f}s")

# TEST 1: Simple
print("")
print("=" * 70)
print("🎨 TEST 1: Paisaje Simple")
print("=" * 70)

start = time.time()
result = pipe(
    prompt="beautiful sunset over mountains, vibrant colors, detailed, masterpiece, high quality",
    negative_prompt="low quality, blurry, bad",
    num_inference_steps=20,
    height=512,
    width=512,
)
elapsed = time.time() - start

output_path = OUTPUT_DIR / "test1_landscape.png"
result.images[0].save(output_path)
print(f"✅ Generado en {elapsed:.2f}s → {output_path}")

# TEST 2: Personaje Femenino
print("")
print("=" * 70)
print("🎨 TEST 2: Personaje Femenino")
print("=" * 70)

start = time.time()
result = pipe(
    prompt="portrait of beautiful woman, long brown hair, green eyes, detailed face, professional photo, masterpiece",
    negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
    num_inference_steps=25,
    height=768,
    width=512,
)
elapsed = time.time() - start

output_path = OUTPUT_DIR / "test2_character.png"
result.images[0].save(output_path)
print(f"✅ Generado en {elapsed:.2f}s → {output_path}")

# TEST 3: Contenido Adulto Artístico
print("")
print("=" * 70)
print("🎨 TEST 3: Desnudo Artístico")
print("=" * 70)

start = time.time()
result = pipe(
    prompt="artistic nude portrait, beautiful woman, professional photography, tasteful lighting, elegant, masterpiece, high quality",
    negative_prompt="low quality, blurry, deformed, bad anatomy",
    num_inference_steps=30,
    height=1024,
    width=768,
)
elapsed = time.time() - start

output_path = OUTPUT_DIR / "test3_artistic_nude.png"
result.images[0].save(output_path)
print(f"✅ Generado en {elapsed:.2f}s → {output_path}")

# TEST 4: Contenido Sexy/Sensual
print("")
print("=" * 70)
print("🎨 TEST 4: Personaje Sexy")
print("=" * 70)

start = time.time()
result = pipe(
    prompt="sexy woman in lingerie, beautiful face, seductive, bedroom, soft lighting, detailed, masterpiece",
    negative_prompt="low quality, blurry, deformed, ugly",
    num_inference_steps=30,
    height=1024,
    width=768,
)
elapsed = time.time() - start

output_path = OUTPUT_DIR / "test4_sexy.png"
result.images[0].save(output_path)
print(f"✅ Generado en {elapsed:.2f}s → {output_path}")

print("")
print("=" * 70)
print("✅ TODAS LAS PRUEBAS COMPLETADAS")
print("=" * 70)
print("")
print(f"📁 Imágenes generadas en {OUTPUT_DIR}:")
print("   1. test1_landscape.png     - Paisaje")
print("   2. test2_character.png     - Personaje femenino")
print("   3. test3_artistic_nude.png - Desnudo artístico")
print("   4. test4_sexy.png          - Personaje sexy")
print("")
print("🎉 ¡Sistema funcionando perfectamente!")
