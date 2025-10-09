# Intelligent Prompting System - Implementation Summary

## 🎯 **OBJETIVO PRINCIPAL**

Crear un sistema inteligente de generación de imágenes fotorrealistas EXPLÍCITAS con:

- **Contenido adulto** (NSFW, explícito, anatómico)
- **Edades 30+** (desde 30 años hasta senectud)
- **Diversidad étnica** (contrarrestar sesgo racial de modelos CivitAI)
- **Parejas/grupos** (2-3 personas, principalmente mujeres)

---

## ✅ **IMPLEMENTADO**

### 1. **CharacterGenerator** (NUEVO - Sistema de Diversidad)

**Archivo**: `ml_lib/diffusion/intelligent/prompting/character_generator.py`

**Funcionalidad**:

- Genera personajes aleatorios con atributos consistentes
- **CRÍTICO**: Contrarresta sesgo racial de modelos
  - 70% mínimo de personajes no-blancos
  - 30% mínimo con piel oscura/media-oscura
  - Pesos de prompt más altos (1.3-1.6) para etnias no-blancas

**Atributos barajados**:

- Edad (30-85 años)
- Etnia (caucásica, latina, asiática, negra, medio-oriente, mestiza)
- Tono de piel (6 niveles: muy claro → oscuro)
- Color de ojos (8 opciones)
- Color de cabello (8 opciones, apropiado para edad)
- Textura de cabello (6 tipos, **consistente con etnia**)
- Tipo de cuerpo (7 opciones)
- Tamaño de pechos (5 opciones)
- Pose (9 opciones, algunas explícitas)
- Escenario (8 locaciones)
- Iluminación (sugerida automáticamente)

**Consistencia étnica automática**:

```yaml
Ejemplo: Mujer negra africana
  ✓ Piel: "dark skin" (peso 1.6)
  ✓ Etnia: "black, african american" (peso 1.5)
  ✓ Cabello: "black hair" + "coily hair, afro textured" (peso 1.4)
  ✓ Rasgos: consistentes con etnia
```

**Uso**:

```python
from ml_lib.diffusion.intelligent.prompting.character_generator import CharacterGenerator

generator = CharacterGenerator()

# Generar personaje diverso
character = generator.generate(enforce_diversity=True)

# Generar personaje con pose explícita
character = generator.generate(explicit_poses_only=True)

# Generar mujer madura (50-60 años)
character = generator.generate(age_range=(50, 60))

# Obtener prompt
prompt = character.to_prompt()
# Resultado ejemplo:
# (40 years old, forties:1.2), (latina, hispanic:1.3), (medium skin, tan:1.3),
# brown eyes, black hair, (wavy hair:1.1), (subtle age lines, mature beauty:1.1),
# curvy, voluptuous, (large breasts:1.2), (lying down, intimate pose:1.2),
# bedroom, lying on bed, (natural window light, soft bedroom lighting:1.1)
```

---

### 2. **Configuraciones YAML Externalizadas**

Todos los atributos y reglas están en archivos YAML editables:

#### `concept_categories.yaml`

- 12 categorías de conceptos para análisis semántico
- **Anatomía EXPLÍCITA**: genitales, pechos, glúteos, zonas íntimas
- **Actividades EXPLÍCITAS**: actos sexuales, posiciones, penetración
- **Edad**: términos específicos por década (30s, 40s, 50s, 60s+)
- **Diversidad**: etnias, tonos de piel, texturas de cabello

#### `lora_filters.yaml`

- **Bloqueados**: anime, cartoon, manga, underage, etc.
- **Prioritarios**: photorealistic, NSFW, mature, anatomy, etc.
- **Pesos de scoring**: priority 25%, anatomy 20%, keywords 25%, tags 20%, popularity 10%

#### `generation_profiles.yaml`

- **Perfiles de edad** (30s, 40s, 50s, 60+): modificadores de CFG/steps
- **Perfiles de grupo** (single, couple, trio): resoluciones óptimas
- **Perfiles de actividad** (intimate_static, explicit_static, sexual_activity): boosts de CFG/steps
- **Presets de detalle** (soft_focus, standard, maximum_detail)
- **Presets de VRAM** (low, medium, high, ultra)

#### `character_attributes.yaml` (NUEVO)

- **Tonos de piel** (6 niveles) con asociaciones étnicas
- **Etnias** (7 grupos) con pesos de prompt
- **Texturas de cabello** (6 tipos) con consistencia étnica
- **Colores de cabello** (8 opciones) con apropiación por edad
- **Tipos de cuerpo** (7 opciones)
- **Tamaños de pechos** (5 opciones)
- **Poses** (9 opciones, 5 explícitas)
- **Escenarios** (8 locaciones) con iluminación sugerida
- **Reglas de consistencia étnica**
- **Objetivos de diversidad** (min 70% no-blanco, min 30% piel oscura)

#### `prompting_strategies.yaml` (NUEVO)

- **Estructura de prompt óptima** (10 pasos ordenados)
- **Estrategias por modelo** (SDXL base, SDXL NSFW finetuned, Pony)
- **Negative prompts** (bloqueadores de estilo, edad, anatomía)
- **Sintaxis de weighting** (guía de pesos)
- **Optimización de edad** (keywords críticos por década)
- **Prompting anatómico** (piel, pechos, genitales, poses)
- **Técnicas fotográficas** (iluminación, cámara, ángulos)
- **Errores comunes** (over-weighting, conflictos, keyword spam)

#### `local_models.yaml` (NUEVO)

- **Checkpoints locales categorizados** (44 modelos)
  - Primarios: `pornmaster_proSDXLV7`, `epicphotogasm_ultimateFidelity`, etc.
  - Por edad: `naturalBeautiesREAL` para 50+, etc.
  - Por cuerpo: `bbwFantasy` para plus-size, `bigasp` para glúteos
- **Estrategia de selección** por edad/etnia/cuerpo
- **Tests prioritarios** para diversidad étnica
- **LoRAs categorizados** (1206 archivos escaneados)
- **Workflow recomendado** (5 pasos)

---

### 3. **ConfigLoader** (Carga Centralizada)

**Archivo**: `ml_lib/diffusion/intelligent/prompting/config_loader.py`

**Funcionalidad**:

```python
from ml_lib.diffusion.intelligent.prompting.config_loader import get_default_config

# Cargar toda la configuración
config = get_default_config()

# Acceso a secciones
concepts = config.concept_categories
blocked_tags = config.blocked_tags
age_profiles = config.age_profiles

# Recargar desde disco
config = reload_config()
```

---

### 4. **PromptAnalyzer** (Actualizado)

**Cambios**:

- Categorías EXPLÍCITAS (anatomía íntima, actividades sexuales)
- Categorías de edad (age_attributes)
- Categorías de detalles físicos (physical_details)
- Detección de parejas/grupos
- TODO: Integrar ConfigLoader

---

### 5. **ParameterOptimizer** (Actualizado para contenido EXPLÍCITO)

**Cambios**:

**CFG Scale**:

- Rango: **8.0 - 15.0** (antes 7.0-12.0)
- Base: **10.5** (antes 9.0)
- Boosts:
  - +2.5 para actos sexuales explícitos
  - +2.5 para anatomía extremadamente detallada (>8 keywords)
  - +2.0 para anatomía muy detallada (>5 keywords)
  - +1.5 para actividades íntimas
  - +1.0 para desnudez/exposición
  - +1.0 para detalles físicos (poros, piel)
  - +0.5 para indicadores de edad

**Steps**:

- Rango: **20 - 80** (antes 15-60)
- Base: **25** (antes 20)
- Boosts:
  - +20 para anatomía extrema
  - +15 para actos sexuales
  - +15 para anatomía muy detallada
  - +35 para complejidad máxima

**Resolución**:

- Detección de grupo (single/couple/trio)
- Aumento automático para detalle anatómico extremo (+12.5%)
- Rangos específicos:
  - Trio: 1280x896
  - Couple horizontal: 1216x832
  - Single portrait: 896x1152
- Caps: min 768x768, max 1536x1536

**Sampler**: Siempre "DPM++ 2M Karras" para fotorrealismo

**Clip Skip**: Siempre **1** (nunca skip para realismo anatómico)

---

### 6. **LoRARecommender** (Actualizado para contenido EXPLÍCITO)

**Cambios**:

**Filtrado**:

- Bloquea anime, cartoon, manga, underage
- Filtra LoRAs antes de scoring

**Scoring actualizado**:

- **25%** Priority tags (photorealistic, NSFW, mature, 30+)
- **20%** Anatomy tags (breasts, genitals, realistic anatomy)
- **25%** Keyword matching
- **20%** Tag matching
- **10%** Popularity (reducido)

**Nuevos métodos**:

- `_filter_blocked_loras()`: Elimina incompatibles
- `_priority_tag_score()`: Score para NSFW/realistic/mature
- `_anatomy_tag_score()`: Score para detalle anatómico

---

## 📂 **ESTRUCTURA DE ARCHIVOS**

```
ml_lib/diffusion/intelligent/prompting/
├── config/
│   ├── concept_categories.yaml           # Categorías semánticas EXPLÍCITAS
│   ├── lora_filters.yaml                 # Filtros LoRA (bloqueados/prioritarios)
│   ├── generation_profiles.yaml          # Perfiles de edad/grupo/actividad
│   ├── character_attributes.yaml         # Atributos para CharacterGenerator ⭐
│   ├── prompting_strategies.yaml         # Best practices de CivitAI/ComfyUI ⭐
│   └── local_models.yaml                 # Modelos locales ComfyUI (44 checkpoints) ⭐
│
├── config_loader.py                      # Cargador centralizado ⭐
├── character_generator.py                # Generador de personajes diversos ⭐
├── prompt_analyzer.py                    # Análisis semántico (Ollama)
├── parameter_optimizer.py                # Optimización CFG/steps/resolution
├── lora_recommender.py                   # Recomendación LoRAs
└── entities.py                           # Dataclasses

examples/diffusion/
├── character_generator_example.py        # Ejemplos de CharacterGenerator ⭐
├── intelligent_prompting_example.py      # Ejemplos de prompting
└── intelligent_memory_example.py         # Ejemplos de memoria

docs/diffusion/
└── intelligent_prompting_SUMMARY.md      # Este archivo ⭐
```

⭐ = Archivos NUEVOS en esta sesión

---

## 🔧 **PRÓXIMOS PASOS CRÍTICOS**

### 1. **Refactorizar componentes para usar ConfigLoader**

**Archivos a modificar**:

- `prompt_analyzer.py`: Cargar `concept_categories.yaml`
- `lora_recommender.py`: Cargar `lora_filters.yaml`
- `parameter_optimizer.py`: Cargar `generation_profiles.yaml`

**Beneficio**: Configuración 100% externalizada, editable sin código

---

### 2. **Crear ModelSelector**

**Nuevo archivo**: `model_selector.py`

**Funcionalidad**:

```python
class ModelSelector:
    def select_checkpoint(
        self,
        character: GeneratedCharacter,
        activity_type: str,
        quality_priority: Priority
    ) -> ModelMetadata:
        """
        Selecciona checkpoint óptimo basado en:
        - Edad del personaje (30s → sensuaxlV2, 60+ → naturalBeautiesREAL)
        - Etnia (no-blanco → realismByStableYogi con boost de peso)
        - Tipo de cuerpo (plus-size → bbwFantasy)
        - Actividad (explícita → pornmaster_proSDXLV7)
        """
```

**Usa**: `local_models.yaml` para mapping

---

### 3. **Crear PromptBuilder**

**Nuevo archivo**: `prompt_builder.py`

**Funcionalidad**:

```python
class PromptBuilder:
    def build_prompt(
        self,
        character: GeneratedCharacter,
        loras: List[LoRARecommendation],
        quality_preset: str = "maximum_detail"
    ) -> Tuple[str, str]:  # (positive_prompt, negative_prompt)
        """
        Construye prompt final combinando:
        1. Character prompt (con pesos étnic os)
        2. LoRA triggers
        3. Quality keywords (de prompting_strategies.yaml)
        4. Negative prompt comprehensivo

        Aplica:
        - Orden óptimo (quality → subject → age → ethnicity → anatomy → ...)
        - Weighting correcto según prompting_strategies.yaml
        - Negative prompt de prompting_strategies.yaml
        """
```

---

### 4. **Integración End-to-End: IntelligentGenerationPipeline**

**Nuevo archivo**: `intelligent_generation_pipeline.py`

**Workflow completo**:

```python
class IntelligentGenerationPipeline:
    def generate_from_concept(
        self,
        user_prompt: Optional[str] = None,  # Si None, genera character aleatorio
        enforce_diversity: bool = True,
        explicit_content: bool = True,
        age_range: Optional[Tuple[int, int]] = None
    ) -> GenerationResult:
        """
        Workflow completo:

        1. Si user_prompt:
           a. Analizar con PromptAnalyzer (Ollama)
           b. Detectar edad/etnia/actividad
        2. Si no user_prompt:
           a. Generar character con CharacterGenerator

        3. Seleccionar checkpoint con ModelSelector
        4. Seleccionar LoRAs con LoRARecommender
        5. Optimizar parámetros con ParameterOptimizer
        6. Construir prompt final con PromptBuilder
        7. Configurar offload con ModelOffloader (memoria)

        Retorna:
          - checkpoint: ModelMetadata
          - loras: List[LoRARecommendation]
          - positive_prompt: str
          - negative_prompt: str
          - cfg_scale: float
          - steps: int
          - width: int
          - height: int
          - sampler: str
          - seed: int
          - offload_config: OffloadConfig
          - character_info: GeneratedCharacter (si generado)
          - reasoning: Dict[str, str]  # Explicaciones de decisiones
        """
```

---

### 5. **Testing de Diversidad Étnica**

**CRÍTICO**: Validar que los modelos locales puedan generar etnias diversas

**Test Plan**:

```python
# Test 1: African American woman, 40s
test_prompts = [
    "(40 years old:1.2), (african american:1.5), (dark skin:1.5), (black woman:1.4), (coily hair:1.4)",
    "(55 years old:1.3), (south asian:1.4), (medium brown skin:1.4), (indian:1.3)",
    "(45 years old:1.2), (latina:1.3), (medium skin:1.3), (hispanic:1.3)"
]

models_to_test = [
    "pornmaster_proSDXLV7",
    "realismByStableYogi_v5XLFP16",
    "naturalBeautiesREAL_nbREAL250629"
]

# Para cada combinación:
# - Generar imagen
# - Evaluar si etnia/edad se respetó
# - Ajustar pesos si necesario
# - Documentar mejores modelos por etnia
```

**Resultado esperado**: Tabla de compatibilidad modelo-etnia

---

### 6. **Documentar LoRAs útiles**

De los **1206 LoRAs** locales:

- Identificar LoRAs fotorrealistas (no Pony)
- Categorizar por función (anatomía, ropa, actividad)
- Agregar a `local_models.yaml`

---

### 7. **Crear sistema de logging/telemetría**

Para aprender qué combinaciones funcionan mejor:

```python
class GenerationLogger:
    def log_generation(
        self,
        character: GeneratedCharacter,
        checkpoint: str,
        loras: List[str],
        params: Dict,
        result_path: str,
        user_rating: Optional[int] = None
    ):
        """
        Log a SQLite database para análisis posterior:
        - ¿Qué combinaciones de checkpoint+ethnicity funcionan?
        - ¿Qué LoRAs mejoran diversidad?
        - ¿Qué parámetros dan mejores resultados?
        """
```

---

## 📊 **MÉTRICAS DE ÉXITO**

### Diversidad Étnica:

- ✅ CharacterGenerator genera ≥70% personajes no-blancos
- ⏳ Modelos locales pueden renderizar etnias diversas (test needed)
- ⏳ Pesos de prompt (1.4-1.6) contrarrestan sesgo efectivamente

### Precisión de Edad:

- ✅ Sistema soporta 30-85 años con keywords específicos
- ⏳ Modelos renderan edades 60+ correctamente (test needed)
- ⏳ Grey hair, age lines aparecen apropiadamente

### Detalle Anatómico:

- ✅ Categorías explícitas completas (genitales, actos sexuales)
- ✅ CFG 8-15 para precisión anatómica
- ✅ Steps 20-80 para detalle extremo
- ⏳ Validar calidad de genitales/anatomía íntima en outputs

### Performance:

- ✅ Sistema de memoria (offloading) implementado
- ✅ Soporta GPUs con <8GB VRAM
- ⏳ Validar tiempos de generación con configuraciones actuales

---

## 🎨 **EJEMPLO DE USO COMPLETO** (cuando esté terminado)

```python
from ml_lib.diffusion.intelligent.prompting.intelligent_generation_pipeline import (
    IntelligentGenerationPipeline
)

# Inicializar pipeline
pipeline = IntelligentGenerationPipeline(
    ollama_url="http://localhost:11434",
    comfyui_checkpoints_dir="/src/ComfyUI/models/checkpoints",
    comfyui_loras_dir="/src/ComfyUI/models/loras"
)

# Generar imagen con personaje aleatorio diverso
result = pipeline.generate_from_concept(
    enforce_diversity=True,
    explicit_content=True,
    age_range=(45, 55)  # Mujeres maduras 45-55
)

print(f"Generated character: {result.character_info.age}y {result.character_info.ethnicity}")
print(f"Selected checkpoint: {result.checkpoint.name}")
print(f"Selected LoRAs: {[l.lora_name for l in result.loras]}")
print(f"Positive prompt: {result.positive_prompt}")
print(f"Negative prompt: {result.negative_prompt}")
print(f"CFG: {result.cfg_scale}, Steps: {result.steps}")
print(f"Resolution: {result.width}x{result.height}")

# Reasoning
print("\nDecision reasoning:")
for decision, reason in result.reasoning.items():
    print(f"  {decision}: {reason}")

# Generar con ComfyUI (integración externa)
# comfyui_client.generate(result.to_comfyui_params())
```

---

## 🔍 **RECURSOS EXTERNOS CONSULTADOS**

- CivitAI model cards (mejores prácticas de prompting NSFW)
- ComfyUI workflows (estructura de prompts fotorrealistas)
- Stable Diffusion community wikis (weighting syntax, CFG ranges)
- SDXL documentation (clip skip, samplers, resolution)

---

## 🚨 **ADVERTENCIAS**

1. **Sesgo racial**: Todos los modelos de CivitAI están sesgados hacia personas blancas. El sistema usa pesos altos (1.4-1.6) para compensar, pero **se requiere testing exhaustivo**.

2. **Pony LoRAs**: Tienes 1206 LoRAs, muchos basados en Pony Diffusion. Pony NO es ideal para fotorrealismo puro. Se recomienda:

   - Buscar/descargar LoRAs SDXL nativos para anatomía realista
   - Testear Pony LoRAs con checkpoints SDXL para ver compatibilidad

3. **Edad 60+**: Modelos tienden a rejuvenecer. Usar:

   - CFG alto (12-15)
   - Age keywords con peso alto (1.3-1.4)
   - Modelo `naturalBeautiesREAL_nbREAL250629` (mejor para senior)

4. **Anatomía explícita**: Requiere:
   - CFG muy alto (13-15)
   - Steps altos (60-80)
   - Checkpoints especializados (`pornmaster_proSDXLV7`)

---

## 💾 **BACKLOG ACTUALIZADO**

Ver: `./docs/backlog/14_intelligent_image_generation/`

**Épica 14** actualizada con:

- ✅ US 14.1: Model Hub Integration (COMPLETADO)
- ✅ US 14.2: Intelligent Prompting System (COMPLETADO + ENHANCED)
- ✅ US 14.3: Memory Management (COMPLETADO)
- 🔄 US 14.4: Pipeline Integration (EN PROGRESO - falta integración final)
- 🆕 US 14.5: Diversity System (NUEVO - CharacterGenerator)
- 🆕 US 14.6: Local Model Integration (NUEVO - ComfyUI models)

**Total horas estimadas**: 150+ horas (original 116h + nuevas features)

---

## 📝 **CONCLUSIÓN**

Se ha implementado un sistema robusto y configurable para generación de contenido adulto fotorrealista EXPLÍCITO con:

- ✅ **Diversidad étnica** garantizada (70% no-blanco minimum)
- ✅ **Edades 30+** con precisión (hasta senectud)
- ✅ **Anatomía explícita** detallada (genitales, actos sexuales)
- ✅ **Configuración externa** (todo en YAML, editable sin código)
- ✅ **Integración con modelos locales** (44 checkpoints, 1206 LoRAs escaneados)
- ✅ **Best practices** de CivitAI/ComfyUI incorporadas

**Falta**:

- Refactorización final para usar ConfigLoader 100%
- ModelSelector implementation
- PromptBuilder implementation
- IntelligentGenerationPipeline end-to-end
- Testing de diversidad étnica con modelos reales

**Prioridad**: TESTING de diversidad étnica con tus modelos locales para validar que los pesos de prompt (1.4-1.6) son suficientes para contrarrestar el sesgo.
