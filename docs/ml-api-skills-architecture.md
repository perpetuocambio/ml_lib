# Arquitectura de Skills para API de ML/IA Generativa en Python

## Estructura del Proyecto

```
ml-api-project/
├── .claude/
│   └── skills/
│       ├── 01-python-code-quality/
│       │   ├── SKILL.md
│       │   ├── linting-config/
│       │   │   ├── .pylintrc
│       │   │   ├── .flake8
│       │   │   └── pyproject.toml
│       │   └── scripts/
│       │       ├── run_quality_checks.py
│       │       └── fix_imports.py
│       │
│       ├── 02-api-design/
│       │   ├── SKILL.md
│       │   ├── templates/
│       │   │   ├── endpoint_template.py
│       │   │   ├── schema_template.py
│       │   │   └── response_template.py
│       │   └── reference/
│       │       ├── rest_best_practices.md
│       │       └── error_handling.md
│       │
│       ├── 03-ml-model-integration/
│       │   ├── SKILL.md
│       │   ├── templates/
│       │   │   ├── model_wrapper.py
│       │   │   ├── inference_pipeline.py
│       │   │   └── batch_processor.py
│       │   ├── reference/
│       │   │   ├── model_optimization.md
│       │   │   └── gpu_management.md
│       │   └── scripts/
│       │       ├── validate_model.py
│       │       └── benchmark_inference.py
│       │
│       ├── 04-diffusion-models/
│       │   ├── SKILL.md
│       │   ├── templates/
│       │   │   ├── stable_diffusion_wrapper.py
│       │   │   ├── image_generation_endpoint.py
│       │   │   └── prompt_processor.py
│       │   ├── reference/
│       │   │   ├── diffusion_parameters.md
│       │   │   ├── prompt_engineering.md
│       │   │   └── safety_filters.md
│       │   └── scripts/
│       │       ├── test_generation.py
│       │       └── optimize_vram.py
│       │
│       ├── 05-testing-ml/
│       │   ├── SKILL.md
│       │   ├── templates/
│       │   │   ├── model_test_template.py
│       │   │   ├── integration_test_template.py
│       │   │   └── performance_test_template.py
│       │   ├── reference/
│       │   │   ├── ml_testing_strategies.md
│       │   │   └── test_data_generation.md
│       │   └── scripts/
│       │       ├── run_ml_tests.py
│       │       └── generate_test_cases.py
│       │
│       ├── 06-async-processing/
│       │   ├── SKILL.md
│       │   ├── templates/
│       │   │   ├── celery_task_template.py
│       │   │   ├── queue_manager.py
│       │   │   └── job_status_tracker.py
│       │   ├── reference/
│       │   │   ├── async_patterns.md
│       │   │   └── queue_configuration.md
│       │   └── scripts/
│       │       └── monitor_queue.py
│       │
│       ├── 07-monitoring-logging/
│       │   ├── SKILL.md
│       │   ├── templates/
│       │   │   ├── structured_logger.py
│       │   │   ├── metrics_collector.py
│       │   │   └── alert_manager.py
│       │   ├── reference/
│       │   │   ├── logging_best_practices.md
│       │   │   ├── metrics_to_track.md
│       │   │   └── alerting_rules.md
│       │   └── scripts/
│       │       ├── analyze_logs.py
│       │       └── generate_report.py
│       │
│       └── 08-security-validation/
│           ├── SKILL.md
│           ├── templates/
│           │   ├── input_validator.py
│           │   ├── rate_limiter.py
│           │   └── auth_middleware.py
│           ├── reference/
│           │   ├── security_checklist.md
│           │   ├── owasp_api_security.md
│           │   └── ml_specific_threats.md
│           └── scripts/
│               ├── security_audit.py
│               └── validate_dependencies.py
```

---

## Skill 1: Python Code Quality

**Archivo**: `.claude/skills/01-python-code-quality/SKILL.md`

```markdown
---
name: Python Code Quality for ML APIs
description: Enforces high-quality Python code standards for machine learning API projects including type hints, linting, formatting, and import organization
---

# Python Code Quality for ML APIs

## Cuando usar este Skill
- Al crear nuevos módulos Python
- Al refactorizar código existente
- Durante revisiones de código
- Antes de commits importantes

## Estándares de calidad

### 1. Type Hints (OBLIGATORIO)
Todos los parámetros de funciones y valores de retorno deben tener type hints:

```python
from typing import List, Dict, Optional, Union
import numpy as np
from pydantic import BaseModel

def process_inference(
    model_input: np.ndarray,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """Procesa inferencia con type hints completos."""
    pass
```

### 2. Docstrings (OBLIGATORIO)
Usa formato Google style para todas las funciones y clases:

```python
def generate_image(
    prompt: str,
    steps: int = 50,
    guidance_scale: float = 7.5
) -> np.ndarray:
    """Genera imagen usando modelo de difusión.
    
    Args:
        prompt: Descripción textual de la imagen a generar
        steps: Número de pasos de difusión (mayor = mejor calidad)
        guidance_scale: Fuerza de adherencia al prompt (típicamente 7-15)
    
    Returns:
        Array NumPy con imagen generada (H, W, C)
    
    Raises:
        ValueError: Si steps < 1 o guidance_scale < 0
        RuntimeError: Si el modelo no está cargado
    
    Examples:
        >>> img = generate_image("a cat in space", steps=30)
        >>> img.shape
        (512, 512, 3)
    """
    pass
```

### 3. Organización de imports
**Orden estricto**:
1. Imports de librería estándar
2. Imports de terceros
3. Imports locales del proyecto

```python
# Estándar
import os
import sys
from typing import List, Dict

# Terceros
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Locales
from app.models.diffusion import StableDiffusionWrapper
from app.utils.image_processing import preprocess_image
```

**Script automático**: Ejecuta `scripts/fix_imports.py` para ordenar automáticamente.

### 4. Longitud de línea
- Máximo: 100 caracteres (configurado en linting-config/)
- Para ML: permitido 120 en líneas con tensores complejos

### 5. Convenciones de nombres

```python
# Constantes: UPPER_SNAKE_CASE
MAX_BATCH_SIZE = 64
DEFAULT_MODEL_PATH = "/models/stable-diffusion"

# Clases: PascalCase
class ImageGenerator:
    pass

# Funciones y variables: snake_case
def process_batch_inference(model_inputs: List[np.ndarray]) -> List[Dict]:
    inference_results = []
    return inference_results

# Variables privadas: _prefijo
class Model:
    def __init__(self):
        self._internal_state = {}
```

### 6. Manejo de errores específico

```python
# MAL: Captura genérica
try:
    result = model.predict(data)
except Exception as e:
    print(f"Error: {e}")

# BIEN: Específico y con contexto
try:
    result = model.predict(data)
except torch.cuda.OutOfMemoryError:
    logger.error(f"CUDA OOM con batch_size={len(data)}")
    raise HTTPException(
        status_code=507,
        detail="Insufficient GPU memory. Try smaller batch size."
    )
except ModelNotLoadedException:
    logger.error("Model not initialized")
    raise HTTPException(status_code=503, detail="Model service unavailable")
```

## Flujo de trabajo de calidad

1. **Antes de codificar**:
   - Define type hints en firma de función
   - Escribe docstring básico

2. **Durante desarrollo**:
   - Ejecuta `scripts/run_quality_checks.py` periódicamente
   - Corrige warnings inmediatamente

3. **Antes de commit**:
   ```bash
   python scripts/run_quality_checks.py --fix
   ```

4. **Revisión automática**:
   - El script valida:
     - pylint (score mínimo: 9.0/10)
     - flake8 (sin errores E, F)
     - mypy (type checking estricto)
     - black (formatting)
     - isort (import ordering)

## Configuraciones incluidas

- **linting-config/.pylintrc**: Configuración pylint para ML
- **linting-config/.flake8**: Reglas flake8 optimizadas
- **linting-config/pyproject.toml**: Configuración black + isort

## Excepciones permitidas

Para código ML específico, puedes usar:
```python
# pylint: disable=too-many-locals
def complex_training_loop():
    # Muchas variables locales son normales en ML
    pass
```

Pero documenta siempre el porqué.
```

---

## Skill 2: API Design

**Archivo**: `.claude/skills/02-api-design/SKILL.md`

```markdown
---
name: RESTful API Design for ML Services
description: Design patterns and best practices for building ML/generative AI REST APIs with FastAPI, including endpoint structure, schemas, and error handling
---

# RESTful API Design for ML Services

## Cuando usar este Skill
- Al diseñar nuevos endpoints
- Al estructurar request/response schemas
- Al implementar versionado de API
- Al manejar errores en endpoints ML

## Principios de diseño

### 1. Estructura de endpoints RESTful

```
POST   /api/v1/images/generate          # Generar nueva imagen
GET    /api/v1/images/{job_id}          # Obtener resultado
GET    /api/v1/images/{job_id}/status   # Verificar estado
DELETE /api/v1/images/{job_id}          # Cancelar/eliminar

POST   /api/v1/models/text/inference    # Inferencia texto
POST   /api/v1/models/vision/classify   # Clasificación imagen

GET    /api/v1/health                   # Health check
GET    /api/v1/models                   # Listar modelos disponibles
```

**Convenciones**:
- Usa sustantivos en plural: `/images`, `/models`
- Usa verbos HTTP correctos: POST (crear), GET (leer), PUT (actualizar), DELETE (eliminar)
- Versiona la API: `/api/v1/`
- Operaciones complejas usan verbos: `/generate`, `/inference`

### 2. Request Schemas con Pydantic

**Template base**: `templates/schema_template.py`

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from enum import Enum

class Scheduler(str, Enum):
    """Schedulers disponibles para difusión."""
    DDIM = "ddim"
    PNDM = "pndm"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"

class ImageGenerationRequest(BaseModel):
    """Request para generación de imágenes."""
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Descripción textual de la imagen a generar",
        examples=["a cat astronaut in space, digital art"]
    )
    
    negative_prompt: Optional[str] = Field(
        None,
        max_length=1000,
        description="Elementos a evitar en la generación"
    )
    
    width: int = Field(
        512,
        ge=256,
        le=1024,
        multiple_of=64,
        description="Ancho de imagen (debe ser múltiplo de 64)"
    )
    
    height: int = Field(
        512,
        ge=256,
        le=1024,
        multiple_of=64,
        description="Alto de imagen (debe ser múltiplo de 64)"
    )
    
    steps: int = Field(
        50,
        ge=1,
        le=150,
        description="Pasos de difusión (más pasos = mejor calidad, más lento)"
    )
    
    guidance_scale: float = Field(
        7.5,
        ge=1.0,
        le=20.0,
        description="Escala de guía CFG (7-10 recomendado)"
    )
    
    scheduler: Scheduler = Field(
        Scheduler.EULER,
        description="Algoritmo de scheduling para difusión"
    )
    
    seed: Optional[int] = Field(
        None,
        ge=0,
        description="Semilla para reproducibilidad (None = aleatoria)"
    )
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        """Valida que dimensiones sean múltiplos de 64."""
        if v % 64 != 0:
            raise ValueError(f"Dimension must be multiple of 64, got {v}")
        return v
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Valida que prompt no esté vacío después de strip."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "a serene mountain landscape at sunset, oil painting",
                "negative_prompt": "people, buildings, modern",
                "width": 768,
                "height": 512,
                "steps": 30,
                "guidance_scale": 8.0,
                "scheduler": "euler",
                "seed": 42
            }
        }
```

### 3. Response Schemas estandarizadas

**Template**: `templates/response_template.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    """Estados posibles de un job asíncrono."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ErrorDetail(BaseModel):
    """Detalle de error estructurado."""
    code: str = Field(..., description="Código de error único")
    message: str = Field(..., description="Mensaje legible")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Información adicional del error"
    )

class BaseResponse(BaseModel):
    """Response base para todas las respuestas."""
    success: bool = Field(..., description="Si la operación fue exitosa")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp ISO 8601 UTC"
    )
    request_id: Optional[str] = Field(
        None,
        description="ID único para tracking"
    )

class JobResponse(BaseResponse):
    """Response para jobs asíncronos."""
    job_id: str = Field(..., description="ID del job creado")
    status: JobStatus = Field(..., description="Estado actual del job")
    estimated_time: Optional[int] = Field(
        None,
        description="Tiempo estimado en segundos"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2025-10-17T10:30:00Z",
                "request_id": "req_abc123",
                "job_id": "job_xyz789",
                "status": "pending",
                "estimated_time": 15
            }
        }

class ImageGenerationResponse(JobResponse):
    """Response específico para generación de imágenes."""
    result_url: Optional[str] = Field(
        None,
        description="URL para descargar imagen (cuando completed)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadatos de generación (seed usado, etc.)"
    )

class ErrorResponse(BaseResponse):
    """Response para errores."""
    success: bool = Field(default=False)
    error: ErrorDetail = Field(..., description="Detalles del error")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "timestamp