# User Story 0.1: Refactorización a Clases con Tipado Fuerte

**Como desarrollador de la biblioteca,** quiero reemplazar el uso de diccionarios por clases tipadas y enums para mejorar la seguridad de tipos, el autocompletado y prevenir errores en tiempo de desarrollo.

## Contexto

Actualmente, algunos módulos usan diccionarios (`Dict[str, Any]`) donde deberían usar clases con tipado fuerte. También hay strings mágicos que deberían ser enums.

## Objetivos

1. Eliminar uso excesivo de diccionarios
2. Convertir strings mágicos en Enums
3. Crear clases dataclass bien tipadas
4. Mejorar la documentación de tipos

## Tareas

### Task 0.1.1: Auditoría del Código Existente

- **Descripción:** Identificar todos los usos de `Dict[str, Any]` y strings mágicos en el código
- **Entregable:** Lista de archivos y líneas que necesitan refactorización
- **Prioridad:** Alta
- **Estimación:** 2 horas

**Checklist:**
- [ ] Buscar todos los `Dict[str, Any]` en la codebase
- [ ] Identificar strings que representan opciones limitadas
- [ ] Listar campos `metadata: Dict[str, Any]`
- [ ] Documentar casos de uso válido de diccionarios

### Task 0.1.2: Refactorizar BaseModel y Core

- **Descripción:** Refactorizar el módulo core para usar clases en lugar de diccionarios
- **Entregable:** Módulo core actualizado con clases tipadas
- **Prioridad:** Alta
- **Estimación:** 4 horas

**Cambios específicos:**

```python
# ANTES (core/models/base_model.py)
@dataclass
class BaseModel:
    name: str
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)  # ❌
    is_fitted: bool = False  # ❌

# DESPUÉS
from enum import Enum, auto

class ModelState(Enum):
    INITIALIZED = auto()
    TRAINING = auto()
    FITTED = auto()
    VALIDATING = auto()
    READY = auto()
    FAILED = auto()

@dataclass
class ModelMetadata:
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    version: str
    tags: list[str] = field(default_factory=list)

@dataclass
class BaseModel:
    name: str
    version: str
    state: ModelState = ModelState.INITIALIZED  # ✅ Enum
    metadata: ModelMetadata  # ✅ Clase tipada
```

**Checklist:**
- [ ] Crear enum `ModelState`
- [ ] Crear clase `ModelMetadata`
- [ ] Refactorizar `BaseModel`
- [ ] Actualizar tests
- [ ] Actualizar documentación

### Task 0.1.3: Crear Enums para Visualización

- **Descripción:** Reemplazar strings mágicos en módulo de visualización con enums
- **Entregable:** Enums para plot types, styles, color schemes
- **Prioridad:** Alta
- **Estimación:** 3 horas

**Cambios específicos:**

```python
# visualization/models/enums.py (nuevo archivo)
from enum import Enum

class PlotType(Enum):
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"

class PlotStyle(Enum):
    DEFAULT = "default"
    SEABORN = "seaborn"
    GGPLOT = "ggplot"
    DARK_BACKGROUND = "dark_background"

class ColorScheme(Enum):
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    COOLWARM = "coolwarm"
```

**Checklist:**
- [ ] Crear archivo `visualization/models/enums.py`
- [ ] Definir `PlotType`, `PlotStyle`, `ColorScheme`
- [ ] Actualizar `PlotConfig` para usar enums
- [ ] Refactorizar servicios de visualización
- [ ] Actualizar tests
- [ ] Actualizar app EcoML Analyzer

### Task 0.1.4: Enums para Linear Algebra

- **Descripción:** Crear enums para operaciones de álgebra lineal
- **Entregable:** Enums para precision, layout, decomposition methods
- **Prioridad:** Alta
- **Estimación:** 3 horas

**Cambios específicos:**

```python
# linalg/models/enums.py (nuevo archivo)
from enum import Enum

class MatrixPrecision(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT128 = "float128"

class MemoryLayout(Enum):
    ROW_MAJOR = "C"
    COLUMN_MAJOR = "F"

class DecompositionMethod(Enum):
    LU = "lu"
    QR = "qr"
    SVD = "svd"
    CHOLESKY = "cholesky"
    EIGEN = "eigen"
    SCHUR = "schur"

class SolverMethod(Enum):
    DIRECT = "direct"
    ITERATIVE = "iterative"
    CONJUGATE_GRADIENT = "cg"
    GMRES = "gmres"
```

**Checklist:**
- [ ] Crear archivo `linalg/models/enums.py`
- [ ] Definir todos los enums necesarios
- [ ] Actualizar `MatrixOperationConfig`
- [ ] Refactorizar servicios
- [ ] Actualizar tests

### Task 0.1.5: Enums para Optimization

- **Descripción:** Crear enums para optimizadores y schedulers
- **Entregable:** Enums para optimizer types, scheduler types, convergence criteria
- **Prioridad:** Alta
- **Estimación:** 4 horas

**Cambios específicos:**

```python
# optimization/models/enums.py (nuevo archivo)
from enum import Enum

class OptimizerType(Enum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    NESTEROV = "nesterov"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"
    ADAMAX = "adamax"
    NADAM = "nadam"
    LBFGS = "lbfgs"
    BFGS = "bfgs"

class SchedulerType(Enum):
    CONSTANT = "constant"
    STEP_DECAY = "step_decay"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARMUP = "warmup"

class ConvergenceCriterion(Enum):
    GRADIENT_NORM = "gradient_norm"
    LOSS_CHANGE = "loss_change"
    PARAMETER_CHANGE = "parameter_change"
    MAX_ITERATIONS = "max_iterations"

@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType  # ✅ Enum
    learning_rate: float
    momentum: float = 0.0
    weight_decay: float = 0.0
```

**Checklist:**
- [ ] Crear archivo `optimization/models/enums.py`
- [ ] Definir todos los enums necesarios
- [ ] Actualizar configuraciones
- [ ] Refactorizar servicios
- [ ] Actualizar tests

### Task 0.1.6: Refactorizar Hyperparameters

- **Descripción:** Convertir `Hyperparameters` de contenedor de dict a clase tipada
- **Entregable:** Clase `Hyperparameters` refactorizada
- **Prioridad:** Media
- **Estimación:** 3 horas

**Cambios específicos:**

```python
# ANTES
@dataclass
class Hyperparameters:
    values: Dict[str, Any] = field(default_factory=dict)  # ❌

# DESPUÉS
@dataclass
class Hyperparameters:
    """Hiperparámetros con validación."""
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 100

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.momentum <= 1:
            raise ValueError("momentum must be in [0, 1]")
        # ... más validaciones
```

**Checklist:**
- [ ] Refactorizar clase `Hyperparameters`
- [ ] Añadir validación en `__post_init__`
- [ ] Crear subclases específicas si necesario
- [ ] Actualizar tests

### Task 0.1.7: Validación de Dict Usage Permitido

- **Descripción:** Documentar y validar usos legítimos de diccionarios
- **Entregable:** Documento con casos de uso válidos
- **Prioridad:** Media
- **Estimación:** 2 horas

**Usos válidos de diccionarios:**
1. Parsing de JSON/API responses
2. Registros dinámicos (plugin systems)
3. kwargs en funciones
4. Cache interno

**Checklist:**
- [ ] Documentar casos válidos
- [ ] Añadir type hints específicos (no `Any`)
- [ ] Usar TypedDict donde sea posible
- [ ] Crear guía de uso

### Task 0.1.8: Crear Tests de Calidad

- **Descripción:** Crear tests que verifiquen calidad de código
- **Entregable:** Suite de tests de calidad
- **Prioridad:** Media
- **Estimación:** 3 horas

**Tests a crear:**
```python
def test_no_bare_dicts_in_public_api():
    """Verifica que la API pública no use Dict[str, Any]."""
    # Escanear módulos y verificar signatures

def test_enums_for_limited_options():
    """Verifica que opciones limitadas usen Enums."""
    # Verificar que no hay strings mágicos

def test_dataclasses_have_validation():
    """Verifica que dataclasses tengan validación."""
    # Verificar __post_init__
```

**Checklist:**
- [ ] Test anti-Dict[str, Any]
- [ ] Test anti-strings mágicos
- [ ] Test validación de dataclasses
- [ ] Integrar en CI/CD

### Task 0.1.9: Actualizar Documentación

- **Descripción:** Actualizar toda la documentación con nuevos tipos
- **Entregable:** Documentación actualizada
- **Prioridad:** Media
- **Estimación:** 2 horas

**Checklist:**
- [ ] Actualizar docstrings
- [ ] Actualizar README
- [ ] Actualizar ejemplos
- [ ] Actualizar tutoriales

### Task 0.1.10: Migration Guide

- **Descripción:** Crear guía de migración para código existente
- **Entregable:** Guía de migración
- **Prioridad:** Baja
- **Estimación:** 2 horas

**Checklist:**
- [ ] Ejemplos de migración
- [ ] Breaking changes documentados
- [ ] Scripts de migración automática
- [ ] Deprecation warnings

## Criterios de Aceptación

- [ ] No hay `Dict[str, Any]` en APIs públicas (excepto casos documentados)
- [ ] Todos los strings de opciones limitadas son Enums
- [ ] Todas las clases tienen validación
- [ ] Tests de calidad pasando
- [ ] Documentación actualizada
- [ ] Type checking sin errores
- [ ] Código legacy migrado o deprecado

## Estimación Total

**28 horas** (~3.5 días de desarrollo)

## Notas

- Esta US debe completarse antes de continuar con nuevos módulos
- Establece los estándares de calidad para todo el proyecto
- Previene deuda técnica futura

## Referencias

- `docs/CODE_QUALITY_GUIDELINES.md`
- Python Dataclasses: https://docs.python.org/3/library/dataclasses.html
- Python Enum: https://docs.python.org/3/library/enum.html
