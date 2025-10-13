# Integration Tests - ML_LIB Diffusion

Estos tests de integración verifican el funcionamiento completo del módulo de difusión con modelos reales y generación de imágenes.

## 📋 Requisitos

### Sistema de Gestión de Paquetes

Este proyecto usa **`uv`** para gestión de dependencias. Si no lo tienes instalado:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dependencias

Las dependencias se instalan automáticamente con el script de setup:

- **PyTorch** (con CUDA 11.8)
- **Diffusers** + Transformers
- **Pillow** + SafeTensors
- **Pytest** + plugins

### Hardware Recomendado

- **CUDA GPU**: RTX 3060+ con 6GB+ VRAM (requerido para velocidad razonable)
- **CPU**: Funciona pero es **extremadamente lento** (~5-10 min por imagen)
- **RAM**: 16GB+ recomendado

### Opcional

- **Ollama**: Para tests con análisis semántico (http://localhost:11434)
  ```bash
  # Instalar Ollama
  curl https://ollama.ai/install.sh | sh

  # Descargar modelo dolphin3 (permite contenido NSFW)
  ollama pull dolphin3

  # Iniciar servidor
  ollama serve
  ```

  **Nota**: Usamos **dolphin3** porque es un modelo sin restricciones que permite analizar y generar contenido NSFW/adulto, necesario para los tests de generación de personajes adultos.

## 🚀 Quick Start

### 1. Setup del Entorno (Primera Vez)

```bash
cd /src/perpetuocambio/ml_lib/tests/integration

# Instalar dependencias con uv
./setup_test_env.sh
```

Esto tomará unos minutos la primera vez (descarga PyTorch, diffusers, etc.)

### 2. Ejecutar Tests

```bash
# Test rápido (1 imagen, ~2-3s en GPU)
./run_tests_uv.sh quick

# Tests básicos
./run_tests_uv.sh basic

# Tests de personajes
./run_tests_uv.sh character

# Tests con Ollama (requiere Ollama corriendo)
./run_tests_uv.sh ollama

# Tests NSFW (contenido adulto)
./run_tests_uv.sh nsfw

# Todos los tests (~10-15 min)
./run_tests_uv.sh all
```

### Usando pytest directamente

```bash
# Todos los tests
pytest -v -s

# Tests específicos
pytest test_image_generation.py -v -s
pytest test_character_generation.py -v -s
pytest test_adult_content_generation.py -v -s -m nsfw

# Solo tests sin Ollama
pytest -v -s -m "not requires_ollama"

# Solo tests NSFW
pytest -v -s -m nsfw
```

## 📁 Estructura de Tests

### `test_image_generation.py`

Tests básicos de generación de imágenes:

- ✅ Generación simple
- ✅ Generación con negative prompt
- ✅ Parámetros customizados
- ✅ Análisis de prompts
- ✅ Integración con Ollama
- ✅ Modos de memoria (balanced/aggressive)

### `test_character_generation.py`

Tests de generación de personajes:

- ✅ Personajes básicos
- ✅ Estilos artísticos (anime, realistic, fantasy)
- ✅ Rasgos de personalidad
- ✅ Generación desde descripción estructurada
- ✅ Variaciones de personajes
- ✅ Generación de party (múltiples personajes)
- ✅ Feedback loop

### `test_adult_content_generation.py`

Tests para contenido adulto (NSFW):

- ✅ Artistic nudes
- ✅ Boudoir photography
- ✅ Pin-up art style
- ✅ Personajes para juegos adultos
- ✅ Escenas románticas
- ✅ Control de calidad anatómica
- ✅ Diferentes tipos de cuerpo
- ✅ Metadata para contenido adulto
- ✅ Sistema de advertencias

## ⚙️ Configuración

### pytest.ini

Configuración de pytest con:
- Markers para categorización
- Timeout de 300s por test
- Logging a consola
- Filtros de warnings

### Markers Disponibles

- `@pytest.mark.slow` - Tests lentos
- `@pytest.mark.nsfw` - Contenido adulto
- `@pytest.mark.requires_gpu` - Requiere GPU
- `@pytest.mark.requires_ollama` - Requiere Ollama
- `@pytest.mark.integration` - Test de integración

## 📊 Salida de Tests

### Imágenes Generadas

Las imágenes se guardan en:
```
/tmp/pytest-<session>/test_<name>/
```

Cada test genera imágenes en su directorio temporal.

### Logs

Los tests muestran:
- ✅ Status de cada test
- 📝 Explicaciones de generación
- ⏱️  Tiempos de generación
- 💾 Uso de VRAM
- 📊 Parámetros utilizados

## 🔧 Troubleshooting

### "torch not available"

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "diffusers not found"

```bash
pip install diffusers transformers accelerate
```

### "CUDA out of memory"

Reduce el tamaño de imagen o usa modo "aggressive":

```python
options = GenerationOptions(
    width=512,
    height=512,
    memory_mode="aggressive"
)
```

### "Ollama connection refused"

Inicia Ollama:

```bash
ollama serve
```

O salta los tests de Ollama:

```bash
pytest -m "not requires_ollama"
```

## 🎯 Ejemplos de Uso

### Test Individual

```bash
# Generar una imagen de test
pytest test_image_generation.py::TestBasicGeneration::test_simple_generation -v -s
```

### Ver Imágenes Generadas

```bash
# Encuentra el directorio de output
ls -la /tmp/pytest-of-*/pytest-current/

# Ver última imagen generada
eog /tmp/pytest-of-*/pytest-current/test_*/test_*.png
```

### Custom Test

```python
def test_my_custom_generation():
    generator = ImageGenerator()
    image = generator.generate_from_prompt(
        prompt="my custom prompt",
        steps=25,
        seed=42
    )
    image.save("/tmp/my_test.png")
```

## ⚠️ Contenido Adulto

Los tests NSFW están marcados con `@pytest.mark.nsfw` y deben ejecutarse explícitamente:

```bash
pytest -m nsfw -v -s
```

**Advertencia**: Estos tests generan imágenes de contenido adulto. Solo ejecutar en entornos apropiados.

## 📈 Métricas Esperadas

### Tiempos de Generación (GPU RTX 3090)

- Imagen 512x512: ~2-3s
- Imagen 768x1024: ~4-6s
- Imagen 1024x1024: ~6-8s

### Uso de Memoria

- Balanced mode: ~6-8GB VRAM
- Aggressive mode: ~4-6GB VRAM

### Calidad

- Los tests verifican que las imágenes sean válidas (PIL Image)
- Los tests verifican resolución correcta
- Inspección visual manual recomendada

## 🐛 Reportar Issues

Si encuentras problemas con los tests:

1. Verifica que todas las dependencias están instaladas
2. Verifica que CUDA está disponible (si es GPU)
3. Revisa los logs del test para detalles
4. Incluye el output completo al reportar

## 📚 Referencias

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)
