# Plan de Reestructuración del Módulo Diffusion

## Problema Actual

La estructura tiene **dos capas** de organización inconsistentes:

```
ml_lib/diffusion/
├── services/          ✅ Estándar (vacío)
├── handlers/          ✅ Estándar (vacío)
├── models/            ✅ Estándar (vacío)
├── config/            ✅ Estándar
├── interfaces/        ✅ Estándar (recién creado)
└── intelligent/       ❌ NO ESTÁNDAR - debe eliminarse
    ├── pipeline/
    │   ├── services/
    │   └── entities/  (debería ser models/)
    ├── memory/
    │   ├── services/
    │   └── entities/  (debería ser models/)
    ├── prompting/
    │   ├── services/
    │   ├── handlers/
    │   ├── models/
    │   └── enums/     (debería estar en models/)
    ├── hub_integration/
    │   ├── services/
    │   └── entities/  (debería ser models/)
    ├── controlnet/
    ├── ip_adapter/
    └── adapters/
```

## Estructura Objetivo

```
ml_lib/diffusion/
├── interfaces/        # Protocolos (ya creado)
├── models/            # Todas las entidades/dataclasses
│   ├── pipeline.py
│   ├── memory.py
│   ├── prompt.py
│   ├── lora.py
│   ├── character.py
│   └── enums/        # Todos los enums centralizados
├── services/          # Lógica de negocio
│   ├── pipeline_orchestrator.py
│   ├── prompt_analyzer.py
│   ├── lora_recommender.py
│   ├── parameter_optimizer.py
│   ├── memory_manager.py
│   ├── model_registry.py
│   ├── character_generator.py
│   └── learning_engine.py
├── handlers/          # Procesadores específicos
│   ├── controlnet_handler.py
│   ├── ip_adapter_handler.py
│   ├── lora_handler.py
│   └── preprocessor_handler.py
├── config/            # Configuración
│   ├── defaults.py
│   └── prompting_config.py  (migrado de YAML)
└── facade.py          # API pública
```

## Mapeo de Archivos

### Modelos (entities → models)

| Origen | Destino |
|--------|---------|
| `intelligent/pipeline/entities/*.py` | `models/pipeline.py` |
| `intelligent/memory/entities/*.py` | `models/memory.py` |
| `intelligent/prompting/entities/*.py` | `models/prompt.py` |
| `intelligent/prompting/models/*.py` | `models/character.py` |
| `intelligent/hub_integration/entities/*.py` | `models/registry.py` |
| `intelligent/prompting/enums/**/*.py` | `models/enums/*.py` |

### Servicios

| Origen | Destino |
|--------|---------|
| `intelligent/pipeline/services/intelligent_pipeline.py` | `services/pipeline_orchestrator.py` |
| `intelligent/prompting/services/prompt_analyzer.py` | `services/prompt_analyzer.py` |
| `intelligent/prompting/services/lora_recommender.py` | `services/lora_recommender.py` |
| `intelligent/prompting/services/parameter_optimizer.py` | `services/parameter_optimizer.py` |
| `intelligent/memory/memory_manager.py` | `services/memory_manager.py` |
| `intelligent/hub_integration/model_registry.py` | `services/model_registry.py` |
| `intelligent/prompting/services/character_generator.py` | `services/character_generator.py` |
| `intelligent/prompting/services/learning_engine.py` | `services/learning_engine.py` |
| `intelligent/memory/services/memory_optimizer.py` | `services/memory_optimizer.py` |

### Handlers

| Origen | Destino |
|--------|---------|
| `intelligent/controlnet/services/controlnet_service.py` | `handlers/controlnet_handler.py` |
| `intelligent/ip_adapter/services/ip_adapter_service.py` | `handlers/ip_adapter_handler.py` |
| `intelligent/controlnet/preprocessors/*.py` | `handlers/preprocessors.py` |
| `intelligent/prompting/handlers/config_loader.py` | `handlers/config_loader.py` |
| `intelligent/prompting/handlers/character_attribute_set.py` | (eliminar - usar dataclasses) |
| `intelligent/adapters/services/adapter_registry.py` | `handlers/adapter_registry.py` |

## Decisiones de Arquitectura

### 1. "entities" vs "models"
**Decisión:** Usar `models/` (estándar Python/Django)
- `entities` es nomenclatura de DDD (Domain Driven Design)
- `models` es más reconocible y estándar

### 2. Enums dispersos
**Decisión:** Centralizar en `models/enums/`
```python
models/enums/
├── __init__.py
├── physical.py    # age, ethnicity, skin_tone, body_type, etc.
├── appearance.py  # clothing, accessories
├── scene.py       # pose, setting, environment
├── style.py       # artistic_style, aesthetic
└── meta.py        # quality, safety, complexity
```

### 3. Config YAML
**Decisión:** Migrar a `config/prompting_config.py` como dataclasses
```python
# config/prompting_config.py
from dataclasses import dataclass

@dataclass
class CharacterAttributeConfig:
    age_ranges: dict
    ethnicities: dict
    skin_tones: dict
    # ... etc

DEFAULT_CHARACTER_CONFIG = CharacterAttributeConfig(...)
```

### 4. Inline imports
**Decisión:** Usar `interfaces/` + dependency injection
- Ya creado en `ml_lib/diffusion/intelligent/interfaces/`
- Mover a `ml_lib/diffusion/interfaces/`
- Los servicios reciben interfaces en `__init__`, no hacen imports internos

## Plan de Ejecución

### Fase 1: Preparación (No rompe nada)
1. ✅ Crear `interfaces/` en raíz diffusion
2. ⬜ Consolidar modelos en `models/`
3. ⬜ Consolidar enums en `models/enums/`
4. ⬜ Migrar configs YAML a `config/prompting_config.py`

### Fase 2: Migración de Servicios
5. ⬜ Refactorizar servicios para usar interfaces
6. ⬜ Mover servicios a `services/`
7. ⬜ Actualizar imports

### Fase 3: Migración de Handlers
8. ⬜ Mover handlers a `handlers/`
9. ⬜ Actualizar imports

### Fase 4: Limpieza
10. ⬜ Eliminar `intelligent/` completo
11. ⬜ Eliminar `config/intelligent_prompting/`
12. ⬜ Actualizar tests
13. ⬜ Actualizar facade

## Validación

Después de cada fase, verificar:
```bash
python3 tests/diffusion/test_facade_simple.py  # Debe pasar
python3 -m pytest tests/diffusion/           # Todos los tests
```

## Beneficios

1. **Estructura clara y predecible** - Un solo nivel, directorios estándar
2. **Imports simples** - `from ml_lib.diffusion.services import PipelineOrchestrator`
3. **Sin referencias circulares** - Gracias a interfaces
4. **Menos anidación** - De 4-5 niveles a 2 niveles
5. **Conforme a estándares Python** - Similar a Django, FastAPI, etc.

## Riesgos

- **Alto impacto**: Afecta ~80 archivos
- **Breaking changes**: Cualquier código externo que importe de `intelligent/` se romperá
- **Tiempo**: Estimado 3-4 horas de refactorización intensiva

## Alternativa Conservadora

Si el riesgo es muy alto, podemos:
1. Mantener `intelligent/` temporalmente
2. Crear estructura nueva en paralelo
3. Hacer facade apuntar a nueva estructura
4. Deprecar `intelligent/` gradualmente
5. Eliminar después de 1-2 versiones

**Recomendación:** Refactorización completa ahora (el módulo no está en producción aún)
