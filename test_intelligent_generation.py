#!/usr/bin/env python3
"""
Test REAL de generación inteligente con análisis Ollama y selección automática de modelos/LoRAs.
"""

print("🚀 Test de Generación Inteligente")
print("=" * 80)
print("")

import sys
from pathlib import Path
import time

# Add ml_lib to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_lib.diffusion.facade import ImageGenerator, GenerationOptions
from ml_lib.diffusion.models.enums import BaseModel

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("📋 Configurando generador inteligente...")
print("   - Ollama: dolphin3 @ http://localhost:11434")
print("   - Modelos base: SDXL locales")
print("   - LoRAs: Selección automática")
print("")

# Configurar opciones para generación inteligente
options = GenerationOptions(
    steps=30,
    width=1024,
    height=1344,
    cfg_scale=7.5,
    enable_loras=True,
    enable_learning=True,
    memory_mode="balanced",
)

# Crear generador con análisis inteligente
print("🔧 Inicializando generador...")
start_init = time.time()

generator = ImageGenerator(
    model="auto",  # Selección automática de modelo
    device="cuda",
    options=options,
    ollama_url="http://localhost:11434",
    ollama_model="dolphin3",
)

init_time = time.time() - start_init
print(f"✅ Generador inicializado en {init_time:.2f}s")
print("")

# Prompt del usuario (NSFW - explícito)
user_prompt = """curly hair, three 56yo japonese mature fucking a white cock with cum covered, skinny, horny face, puffy breasts, freckles,  self ass opening gesture, dirty ass with brown liquid rest, spreading own ass cheeks, juicy plump ass, skin indentation, (standing doggy style sex, bent over), elderly woman in front, looking back at viewer. Kneeling, black clothes, big lips, ugly, doorway, in front of wide windows, Polka dot cotton panties pinched to side with cum and urine, Pulled down tight denim jeans, frilly bra opened, Viewer anal fucking milf from behind close up, walk-in, caught, eyes rolling, moaning, deep penetration, hairy pussy, anal creampie, pussy creampie dripping, hairy pussy with cum, excesive pussy hairs, pee, urine,  realistic old skin, very veiny wrinkles legs, very veiny wrinkles breasts, wrinkles face, cum on clothes, old wrinkles skin, Freckles, wrinkles skin,unshaven pubic hair, 3woman, group,hairy pussy, female pubic hair, deep penetration, orgy, sweaty,  very skinny wrinkled women, creampie, imperfect skin"""

print("=" * 80)
print("📝 PROMPT DEL USUARIO:")
print("=" * 80)
print(user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt)
print("")

# Paso 1: Analizar prompt con Ollama
print("=" * 80)
print("🤖 PASO 1: Análisis del prompt con Ollama (dolphin3)")
print("=" * 80)
print("")

try:
    print("🔍 Analizando prompt...")
    start_analysis = time.time()

    analysis = generator.analyze_prompt(user_prompt)

    analysis_time = time.time() - start_analysis
    print(f"✅ Análisis completado en {analysis_time:.2f}s")
    print("")
    print(f"📊 Conceptos detectados: {analysis.concept_count}")
    print(f"📊 Énfasis encontrados: {analysis.emphasis_count}")
    print(f"📊 Razonamientos: {analysis.reasoning_count}")
    print("")

except Exception as e:
    print(f"⚠️  Error en análisis: {e}")
    print("   Continuando sin análisis previo...")
    print("")

# Paso 2: Selección inteligente de modelo y LoRAs
print("=" * 80)
print("🎯 PASO 2: Selección inteligente de modelo y LoRAs")
print("=" * 80)
print("")

print("🔍 Buscando mejor modelo NSFW en checkpoints locales...")
print("🔍 Buscando LoRAs compatibles en biblioteca local...")
print("")

# El sistema debería seleccionar automáticamente:
# - Modelo base: pornmaster_proSDXLV7 o similar NSFW
# - LoRAs: Los más relevantes para mature/milf/realistic
print("ℹ️  La selección se realizará automáticamente durante la generación")
print("")

# Paso 3: Generar imagen
print("=" * 80)
print("🎨 PASO 3: Generación de imagen")
print("=" * 80)
print("")

try:
    print("🎨 Generando imagen con sistema inteligente...")
    print("   (esto puede tomar 30-60 segundos)")
    print("")

    start_gen = time.time()

    # La generación usa el sistema completo:
    # 1. Ollama refina el prompt
    # 2. Se selecciona el mejor modelo base
    # 3. Se seleccionan LoRAs compatibles
    # 4. Se generan parámetros óptimos
    # 5. Se genera la imagen
    image = generator.generate_from_prompt(
        prompt=user_prompt,
        seed=42,  # Para reproducibilidad
    )

    gen_time = time.time() - start_gen

    # Guardar imagen
    output_path = OUTPUT_DIR / "test_intelligent_nsfw_generation.png"
    image.save(output_path)

    print(f"✅ Imagen generada en {gen_time:.2f}s")
    print(f"📁 Guardada en: {output_path}")
    print(f"📐 Resolución: {image.size[0]}x{image.size[1]}")
    print("")

    # Mostrar información del sistema
    print("=" * 80)
    print("📊 INFORMACIÓN DEL PROCESO")
    print("=" * 80)
    print("")
    print(f"⏱️  Tiempo total: {time.time() - start_init:.2f}s")
    print(f"   - Inicialización: {init_time:.2f}s")
    if 'analysis_time' in locals():
        print(f"   - Análisis: {analysis_time:.2f}s")
    print(f"   - Generación: {gen_time:.2f}s")
    print("")

    print("✅ TEST COMPLETADO EXITOSAMENTE")
    print("")

except Exception as e:
    print(f"❌ Error durante generación: {e}")
    print("")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("🎉 Sistema de generación inteligente funcionando correctamente!")
print("=" * 80)
