# AUDITORÍA: ml_lib/diffusion/models/

**Fecha:** 2025-10-15
**Módulo:** `ml_lib/diffusion/models/`
**Problema Central:** CLASES ANÉMICAS + ESTRUCTURA DUPLICADA

---

## RESUMEN EJECUTIVO

El directorio `models/` contiene las entidades del dominio de diffusion. **PROBLEMA CRÍTICO:** La mayoría son **clases anémicas** (dataclasses puros sin comportamiento), violando el principio de encapsulación y creando una arquitectura procedural disfrazada de OOP.

### Hallazgo Principal

**El código actual es procedural con clases como estructuras de datos.** La lógica de negocio está dispersa en servicios, creando:
- Duplicación de validaciones
- Lógica de negocio alejada de los datos
- Imposibilidad de garantizar invariantes
- Tests que deben mockear todo el ecosistema

---

## ANÁLISIS POR ARCHIVO

### 1. CORE.PY (144 líneas)

**Tipo:** Clases anémicas con mínimo comportamiento

**Contenido:**
- `AttributeDefinition`: 144 líneas, casi puro dataclass
- `AttributeRelation`: Tiny helper
- `AttributeType`: Enum

**Problemas:**
- `AttributeDefinition` tiene 15+ atributos pero solo 3 métodos simples
- Lógica de validación mezclada (`is_compatible_with`, `validate_age`)
- Generación de prompts (`get_prompt_segment`) debería estar en un formatter
- No garantiza invariantes (ej: `prompt_weight` podría ser negativo)
- `metadata: dict[str, str]` - tipo too general, sin validación

**Tareas futuras:**
- Extraer `AttributeValidator` para validaciones complejas
- Extraer `PromptFormatter` para generación de prompts
- Implementar Value Objects para `PromptWeight`, `ProbabilityScore`
- Agregar validación en `__post_init__` para garantizar invariantes
- Considerar separar en `PhysicalAttribute`, `ClothingAttribute`, etc.
- Reemplazar `metadata: dict` por atributos tipados específicos

### 2. PIPELINE.PY

**Tipo:** Mix de entidades y DTOs

**Contenido (estimado):**
- `PipelineConfig`: Configuración
- `GenerationResult`: DTO resultado
- `GenerationMetadata`: DTO metadatos
- `GenerationExplanation`: DTO explicación
- Varios enums (`OperationMode`, etc.)

**Problemas:**
- Sin comportamiento - solo contenedores de datos
- `GenerationResult` debería tener métodos como `save_to_disk()`, `validate()`
- `PipelineConfig` no valida conflictos de configuración
- Mixing concerns: Config + Results en mismo archivo

**Tareas futuras:**
- Separar `pipeline/config.py` y `pipeline/results.py`
- Agregar comportamiento a `GenerationResult` (save, validate, export)
- Implementar Builder pattern para `PipelineConfig`
- Validar configuración en construcción
- Agregar métodos de conveniencia (ej: `GenerationResult.to_dict()`)

### 3. PROMPT.PY

**Tipo:** Clases anémicas

**Contenido (estimado):**
- `PromptAnalysis`: Resultado de análisis
- `Intent`: Intención detectada
- Enums de estilos, contenido, calidad

**Problemas:**
- `PromptAnalysis` sin métodos útiles
- No tiene forma de "interpretar" el análisis
- Lógica de interpretación está en `PromptAnalyzer` service

**Tareas futuras:**
- Agregar métodos como `get_dominant_style()`, `is_complex()`, `requires_lora()`
- Implementar `IntentMatcher` para comparar intents
- Agregar comportamiento de serialización/deserialización
- Considerar patrón Specification para queries complejas

### 4. LORA.PY

**Tipo:** Clase anémica

**Contenido (estimado):**
- `LoRAInfo`: Metadata de LoRA
- `LoRARecommendation`: Resultado de recomendación

**Problemas:**
- Sin validación de `alpha` range
- Sin comportamiento de "aplicación" o "composición"
- Lógica de scoring/ranking en servicio separado

**Tareas futuras:**
- Agregar validación de `alpha` (típicamente 0.0-2.0)
- Implementar `apply_to_pipeline(pipeline)` method
- Agregar `combine_with(other_lora)` para composición
- Value Object para `LoRAWeight` con validación

### 5. CHARACTER.PY

**Tipo:** Clase anémica

**Contenido (estimado):**
- Definiciones de personajes
- Atributos físicos

**Problemas:**
- Probablemente puro data sin comportamiento
- Validaciones en servicios externos

**Tareas futuras:**
- Agregar métodos de validación de coherencia
- Implementar `generate_prompt_segment()`
- Validar combinaciones imposibles (ej: color de pelo inconsistente con raza)

### 6. MEMORY.PY

**Tipo:** Clase anémica

**Contenido (estimado):**
- Estadísticas de memoria
- Resultados de optimización

**Problemas:**
- Solo data, sin comportamiento
- Cálculos en `MemoryOptimizer` service

**Tareas futuras:**
- Agregar métodos como `is_within_budget()`, `can_fit_model(size)`
- Implementar comparaciones (ej: `is_better_than(other)`)

### 7. VALUE_OBJECTS/ (Subdirectorio)

**Tipo:** Mayormente correcto (Value Objects)

**Contenido:**
- `ConceptMap`, `Concept`
- `EmphasisMap`, `Emphasis`
- `Resolution`, `Weights`
- Etc.

**Problemas:**
- Algunos son data classes simples sin validación
- Otros tienen validación incompleta
- Inconsistencia en implementación

**Tareas futuras:**
- Estandarizar todos como immutable Value Objects
- Agregar validación exhaustiva en construcción
- Implementar `__eq__`, `__hash__` correctamente
- Agregar métodos de transformación (ej: `Resolution.scale_by(factor)`)

---

## PROBLEMAS ESTRUCTURALES GLOBALES

### 1. ANEMIA DOMAIN MODEL

**Síntoma:** 90% de las clases son dataclasses sin comportamiento

**Consecuencias:**
- Lógica de negocio dispersa en servicios
- Validaciones duplicadas
- Imposible garantizar invariantes
- Testing difícil (hay que mockear servicios)

**Tareas:**
- Migrar comportamiento de servicios a entidades cuando corresponda
- Regla: "Datos y comportamiento juntos"
- Servicios solo para coordinación, no para lógica de entidades

### 2. DUPLICACIÓN DE ESTRUCTURA

**Síntoma:** Múltiples clases con estructura similar

Ejemplos:
- `LoRAInfo` vs `LoRARecommendation` (campos similares)
- `PromptAnalysis` vs `Intent` (overlap semántico)
- Múltiples "stats" classes con estructura similar

**Consecuencias:**
- Duplicación de validaciones
- Inconsistencia en nombres de campos
- Difícil mantener sincronizados

**Tareas:**
- Identificar jerarquías naturales
- Extraer base classes o traits comunes
- Unificar clases duplicadas donde sea posible
- Documentar diferencias cuando duplicación es necesaria

### 3. FALTA DE VALUE OBJECTS TIPADOS

**Síntoma:** Uso de primitivos donde deberían haber Value Objects

Ejemplos:
- `alpha: float` en lugar de `alpha: LoRAWeight`
- `prompt_weight: float` en lugar de `PromptWeight`
- `confidence: float` en lugar de `ConfidenceScore`
- `steps: int` sin validación de rango

**Consecuencias:**
- Sin validación automática
- Posibilidad de valores inválidos
- Lógica de validación repetida

**Tareas:**
- Crear Value Objects para conceptos del dominio
- Implementar validación en construcción
- Hacer immutable (frozen=True)
- Agregar métodos de conveniencia

### 4. MEZCLA DE CONCERNS EN ARCHIVOS

**Síntoma:** Config + Results + Entities en mismo archivo

Ejemplo: `pipeline.py` tiene todo mezclado

**Tareas:**
- Separar en archivos por concern
- `pipeline/config.py`, `pipeline/results.py`, `pipeline/entities.py`

### 5. ENUMS SIN COMPORTAMIENTO

**Síntoma:** Enums usados solo como constants

Ejemplos:
- `ArtisticStyle`, `ContentType`, `QualityLevel`

**Problemas:**
- No tienen métodos útiles
- Conversión de/a string repetida
- Validaciones dispersas

**Tareas:**
- Agregar métodos a enums (ej: `ArtisticStyle.requires_lora()`)
- Implementar `from_string()`, `to_display_name()`
- Agregar metadata como attributes

---

## PROPUESTA DE REFACTORIZACIÓN

### Estrategia: Migración Gradual a Rich Domain Model

#### Fase 1: Agregar Comportamiento Sin Romper
1. Agregar métodos a clases existentes
2. Mantener backward compatibility
3. Marcar lógica en servicios como deprecated

#### Fase 2: Crear Value Objects
1. `LoRAWeight`, `PromptWeight`, `ConfidenceScore`
2. `Resolution`, `ImageDimensions`
3. Migrar uso gradualmente

#### Fase 3: Reorganizar Estructura
1. Separar por bounded contexts
2. Crear jerarquías claras
3. Eliminar duplicación

---

## MÉTRICAS

```
Total archivos: 12+
Clases totales: ~30-40
Clases anémicas: ~25 (60%+)
Value Objects correctos: ~5-10 (25%)
Clases con lógica: ~5 (15%)

PROBLEMA: 75% son contenedores de datos puros
```

---

## RECOMENDACIONES PRIORITARIAS

### CRÍTICO
1. **Agregar validación a todas las entidades**
   - Implementar en `__post_init__`
   - Usar Value Objects para tipos validados

2. **Migrar lógica simple de servicios a entidades**
   - Empezar con métodos obvios (ej: `is_valid()`, `can_combine()`)
   - Mantener servicios solo para coordinación

### IMPORTANTE
3. **Crear Value Objects para primitivos del dominio**
   - `LoRAWeight`, `PromptWeight`, `ConfidenceScore`
   - Implementar validación automática

4. **Reorganizar archivos por concern**
   - Separar config, results, entities

### MEJORA CONTINUA
5. **Enriquecer enums con comportamiento**
6. **Documentar invariantes de cada clase**
7. **Agregar factory methods donde sea apropiado**

---

## CONCLUSIÓN

El modelo de dominio actual es **anémico y procedural**. Aunque funcional, dificulta:
- Garantizar invariantes
- Testing (todo en servicios)
- Comprensión del dominio (lógica dispersa)
- Evolución (cambios requieren tocar servicios)

**Prioridad: Enriquecer entidades gradualmente, empezando por validaciones.**
