# ML Library - Contexto de Proyecto para Claude Code

## Descripción General

ML Library es una biblioteca de Machine Learning de alto rendimiento y código agnóstico escrita en Python, diseñada con arquitectura modular, tipado estricto y patrones de diseño orientados a interfaces.

### Visión del Proyecto

Construir una biblioteca ML moderna que combine:

- **Tipado estricto** con Python type hints para seguridad en desarrollo
- **Arquitectura modular** siguiendo principios SOLID
- **Alto rendimiento** optimizado con NumPy y bibliotecas de bajo nivel
- **Extensibilidad** mediante interfaces y sistema de plugins
- **Código agnóstico** para trabajar con diferentes backends

## Arquitectura Modular Estricta

### Principios de Estructura

El proyecto sigue una arquitectura modular estricta validada por `scripts/check_module_structure.py`:

**Reglas de Estructura:**

1. ❌ **No ficheros .py en la raíz del proyecto** (todo debe estar en `ml_lib/`)
2. ❌ **No ficheros .py en la raíz de un módulo** (excepto `__init__.py`)
3. ✅ **Ficheros .py solo en subdirectorios:** `services/`, `interfaces/`, `models/`, `handlers/`

### Patrón de Organización Modular

Cada módulo en `ml_lib/` sigue la siguiente estructura:

```
ml_lib/[modulo]/
├── __init__.py                 # Exporta API pública del módulo
├── interfaces/                 # Interfaces (ABC) y Protocols
│   ├── __init__.py
│   └── [nombre]_interface.py
├── models/                     # Data models (dataclasses, TypedDict)
│   ├── __init__.py
│   └── [nombre]_model.py
├── services/                   # Lógica de negocio y algoritmos
│   ├── __init__.py
│   └── [nombre]_service.py
└── handlers/                   # Manejo de errores, config, cache
    ├── __init__.py
    └── [nombre]_handler.py
```

### Responsabilidades por Capa

**Interfaces (`interfaces/`):**

- Definición de contratos con ABC (Abstract Base Class)
- Uso de `typing.Protocol` para duck typing cuando sea apropiado
- Generics con TypeVar para tipado flexible
- Sin lógica de implementación

**Modelos (`models/`):**

- Data classes con `@dataclass`
- TypedDict para estructuras de datos
- Validación de tipos con type hints
- Sin lógica de negocio

**Servicios (`services/`):**

- Implementación de algoritmos
- Lógica de negocio principal
- Operaciones computacionales
- Inyección de dependencias

**Handlers (`handlers/`):**

- Manejo de errores específicos
- Gestión de configuración
- Cache y optimización de memoria
- Validación y transformación de datos

## Módulos del Proyecto

### 🏗️ Core Infrastructure (Implementado)

**`ml_lib/core/`** - Componentes fundamentales

- ✅ Interfaces base: `EstimatorInterface`, `TransformerInterface`, `MetricInterface`, `OptimizerInterface`
- ✅ Servicios: `ValidationService`, `LoggingService`
- ✅ Handlers: `ErrorHandler`, `ConfigHandler`
- ✅ Modelos: `BaseModel`, `Metadata`

### 🔢 Linear Algebra (Implementado)

**`ml_lib/linalg/`** - Operaciones de álgebra lineal optimizadas

- ✅ Interfaces: `MatrixOperationInterface`, `DecompositionInterface`, `SolverInterface`, `BLASInterface`, `LAPACKInterface`
- ✅ Modelos: `Matrix`, `SparseMatrix`, `DecompositionResult`, `LinearSystemSolution`, `EigenDecomposition`
- ✅ Servicios: `BLASService`, `LAPACKService`, `MatrixOperationService`, `DecompositionService`, `SolverService`, `SparseMatrixService`
- ✅ Handlers: `LinearAlgebraErrorHandler`, `MatrixConfigHandler`, `MemoryLayoutHandler`, `PrecisionHandler`

### 📊 Visualization (Implementado)

**`ml_lib/visualization/`** - Componentes de visualización generales

- ✅ Interfaces: `VisualizationInterface`, `PlotTypeInterface`
- ✅ Modelos: `PlotConfig`, `VisualizationMetadata`
- ✅ Servicios: `VisualizationService`, `PlottingService`
- ✅ Handlers: `VisualizationErrorHandler`, `ImageExportHandler`, `VisualizationConfigHandler`
- Diseñado para ser agnóstico al dominio y reutilizable

### 🔄 Automatic Differentiation (En Desarrollo)

**`ml_lib/autograd/`** - Diferenciación automática

- Interfaces: `DifferentiableInterface`, `OperationInterface`, `VariableInterface`
- Modelos: `ComputationalGraph`, `Variable`, `OperationNode`
- Servicios: `GraphBuilderService`, `GradientComputationService`, `BackwardService`
- Handlers: `NodeHandler`, `OperationHandler`, `TapeHandler`

### 🎯 Optimization (En Desarrollo)

**`ml_lib/optimization/`** - Algoritmos de optimización numérica

- Interfaces: `OptimizerInterface`, `SchedulerInterface`, `ConstraintInterface`
- Modelos: `OptimizerState`, `OptimizationResult`, `ConvergenceCriteria`
- Servicios: `FirstOrderOptimizerService`, `SecondOrderOptimizerService`, `LineSearchService`
- Handlers: `GradientHandler`, `MomentumHandler`, `LearningRateHandler`

### 🌐 Kernel Methods (Planificado)

**`ml_lib/kernels/`** - Métodos de kernel y SVM

- Interfaces: `KernelInterface`, `KernelMethodInterface`, `SimilarityInterface`
- Modelos: `KernelMatrix`, `KernelParams`, `SVMModel`
- Servicios: `KernelComputationService`, `KernelMatrixService`, `HyperparameterService`
- Handlers: `KernelCacheHandler`, `GramMatrixHandler`

### 📈 Probabilistic Models (Planificado)

**`ml_lib/probabilistic/`** - Modelos probabilísticos

- Interfaces: `DistributionInterface`, `GraphicalModelInterface`, `InferenceInterface`
- Modelos: `BayesianNetwork`, `MarkovChain`, `LatentVariableModel`
- Servicios: `InferenceService`, `SamplingService`, `EMService`
- Handlers: `DistributionHandler`, `GibbsHandler`, `VariationalHandler`

### 🧠 Neural Networks (Planificado)

**`ml_lib/neural/`** - Redes neuronales

- Interfaces: `LayerInterface`, `ActivationInterface`, `LossInterface`
- Modelos: `NeuralNetwork`, `LayerConfig`, `TrainingState`
- Servicios: `LayerService`, `ActivationService`, `BackpropagationService`
- Handlers: `WeightInitializationHandler`, `ForwardPassHandler`, `RegularizationHandler`

### 🌲 Ensemble Methods (Planificado)

**`ml_lib/ensemble/`** - Métodos de ensemble

- Interfaces: `EnsembleInterface`, `WeakLearnerInterface`, `AggregationInterface`
- Modelos: `EnsembleModel`, `DecisionTree`, `BoostingState`
- Servicios: `BoostingService`, `BaggingService`, `StackingService`
- Handlers: `TreeBuilderHandler`, `VotingHandler`, `MetaLearnerHandler`

### 🔧 Feature Engineering (Planificado)

**`ml_lib/feature_engineering/`** - Ingeniería de características

- Interfaces: `SelectorInterface`, `ExtractorInterface`, `FeatureInterface`
- Modelos: `FeatureSet`, `TransformationPipeline`, `FeatureMetadata`
- Servicios: `SelectionService`, `ExtractionService`, `SynthesisService`
- Handlers: `ImportanceHandler`, `TransformationHandler`, `InteractionHandler`

### 📦 Data Processing (Planificado)

**`ml_lib/data_processing/`** - Procesamiento de datos a escala

- Interfaces: `DataLoaderInterface`, `ProcessorInterface`, `IteratorInterface`
- Modelos: `Dataset`, `Batch`, `DataConfig`
- Servicios: `StreamingService`, `BatchService`, `DistributedService`
- Handlers: `ChunkHandler`, `MemoryMapHandler`, `ParallelHandler`

### 🎲 Uncertainty Quantification (Planificado)

**`ml_lib/uncertainty/`** - Cuantificación de incertidumbre

- Interfaces: `UncertaintyInterface`, `CalibratorInterface`, `IntervalInterface`
- Modelos: `UncertaintyEstimate`, `CalibrationCurve`, `PredictionInterval`
- Servicios: `CalibrationService`, `ConformalService`, `EnsembleUncertaintyService`
- Handlers: `PredictionIntervalHandler`, `DropoutHandler`, `TemperatureHandler`

### ⏱️ Time Series (Planificado)

**`ml_lib/time_series/`** - Modelado de series temporales

- Interfaces: `ForecasterInterface`, `TimeSeriesModelInterface`, `SequenceInterface`
- Modelos: `TimeSeries`, `ForecastResult`, `ARIMAModel`
- Servicios: `ForecastingService`, `DecompositionService`, `StationarityService`
- Handlers: `SeasonalityHandler`, `TrendHandler`, `ResidualHandler`

### 🎮 Reinforcement Learning (Planificado)

**`ml_lib/reinforcement/`** - Aprendizaje por refuerzo

- Interfaces: `AgentInterface`, `EnvironmentInterface`, `PolicyInterface`
- Modelos: `Agent`, `State`, `Transition`
- Servicios: `PolicyService`, `ValueFunctionService`, `EnvironmentService`
- Handlers: `ReplayBufferHandler`, `ExplorationHandler`, `RewardHandler`

### 🔍 Interpretability (Planificado)

**`ml_lib/interpretability/`** - Interpretación de modelos

- Interfaces: `ExplainerInterface`, `AttributionInterface`, `VisualizationInterface`
- Modelos: `Explanation`, `AttributionMap`, `FeatureImportance`
- Servicios: `ExplanationService`, `AttributionService`, `VisualizationService`
- Handlers: `LIMEHandler`, `SHAPHandler`, `ImportanceHandler`

### 🤖 AutoML (Planificado)

**`ml_lib/automl/`** - Automatización de ML

- Interfaces: `OptimizerInterface`, `SearchSpaceInterface`, `ObjectiveInterface`
- Modelos: `SearchSpace`, `Trial`, `OptimizationResult`
- Servicios: `HyperparameterOptimizationService`, `NASService`, `MetaLearningService`
- Handlers: `TrialHandler`, `BayesianOptimizationHandler`, `ArchitectureSearchHandler`

### ⚖️ Fairness (Planificado)

**`ml_lib/fairness/`** - Equidad y sesgo

- Interfaces: `FairnessMetricInterface`, `DebiaserInterface`, `ConstraintInterface`
- Modelos: `FairnessReport`, `ProtectedAttribute`, `MitigationResult`
- Servicios: `BiasDetectionService`, `MitigationService`, `MetricService`
- Handlers: `DemographicHandler`, `AdversarialDebiasingHandler`, `ConstraintHandler`

### 🚀 Deployment (Planificado)

**`ml_lib/deployment/`** - Despliegue de modelos

- Interfaces: `ServerInterface`, `MonitorInterface`, `RegistryInterface`
- Modelos: `ModelArtifact`, `MonitoringMetrics`, `DeploymentConfig`
- Servicios: `ServingService`, `MonitoringService`, `VersioningService`
- Handlers: `InferenceHandler`, `DriftDetectionHandler`, `ModelRegistryHandler`

### 🔌 Plugin System (Planificado)

**`ml_lib/plugin_system/`** - Sistema de plugins

- Interfaces: `PluginInterface`, `HookInterface`, `ExtensionInterface`
- Modelos: `PluginMetadata`, `HookSpecification`, `ExtensionConfig`
- Servicios: `DiscoveryService`, `LoadingService`, `RegistryService`
- Handlers: `EntryPointHandler`, `HookHandler`, `CallbackHandler`

### ⚡ Performance (Planificado)

**`ml_lib/performance/`** - Rendimiento y optimización

- Interfaces: `ProfilerInterface`, `CompilerInterface`, `CacheInterface`
- Modelos: `ProfilingResult`, `PerformanceMetrics`, `CacheConfig`
- Servicios: `ProfilingService`, `CompilationService`, `CachingService`
- Handlers: `MemoryProfilerHandler`, `GPUHandler`, `JITHandler`

### 🛠️ Utils (Planificado)

**`ml_lib/utils/`** - Utilidades generales

- Interfaces: `SerializableInterface`, `RandomStateInterface`, `ParallelInterface`
- Modelos: `Config`, `RandomState`, `JobConfig`
- Servicios: `SerializationService`, `RandomService`, `ParallelService`
- Handlers: `PickleHandler`, `ThreadPoolHandler`, `ProcessPoolHandler`

## Aplicación de Demostración

### EcoML Analyzer

**`ecoml_analyzer/`** - Aplicación de análisis ecológico

Aplicación completa que demuestra el uso de la biblioteca en un contexto real:

- Análisis de abundancia de especies
- Análisis de diversidad y comunidades ecológicas
- Distribución de especies
- Visualización de resultados ecológicos

Ejemplifica cómo usar los componentes generales (especialmente visualización) en un dominio específico.

## Patrones de Diseño Aplicados

### 1. Interface-Based Design

```python
# Interface (ABC)
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

X = TypeVar("X", bound=np.ndarray)
Y = TypeVar("Y", bound=np.ndarray)

class EstimatorInterface(ABC, Generic[X, Y]):
    @abstractmethod
    def fit(self, X: X, y: Y, **kwargs) -> "EstimatorInterface[X, Y]":
        pass

    @abstractmethod
    def predict(self, X: X) -> Y:
        pass
```

### 2. Service-Based Architecture

```python
# Service con inyección de dependencias
class CustomEstimator(EstimatorInterface):
    def __init__(
        self,
        validation_service: ValidationService,
        error_handler: ErrorHandler
    ):
        self.validation = validation_service
        self.error_handler = error_handler

    def fit(self, X, y, **kwargs):
        self.validation.validate_input_shape(X, 2)
        self.validation.validate_same_length(X, y)
        return self._fit_impl(X, y, **kwargs)
```

### 3. Handler Pattern

```python
# Handler para aspectos transversales
class ErrorHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def handle_execution_error(self, func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
```

### 4. Factory Pattern

```python
# Factory para creación de objetos complejos
class VisualizationFactory:
    @staticmethod
    def create_visualization(config: PlotConfig) -> VisualizationInterface:
        return GeneralVisualization(config)
```

## Stack Tecnológico

### Dependencias Core

- **Python**: ≥3.10
- **NumPy**: ≥1.21.0 (operaciones numéricas)
- **SciPy**: ≥1.7.0 (algoritmos científicos)
- **Pandas**: ≥1.3.0 (manipulación de datos)

### Visualización

- **Matplotlib**: ≥3.5.0
- **Seaborn**: ≥0.11.0
- **Plotly**: ≥5.0.0

### Desarrollo

- **pytest**: Testing
- **mypy**: Type checking
- **black**: Formatting
- **flake8**: Linting
- **isort**: Import sorting

## Guías de Implementación

### Crear un Nuevo Módulo

1. **Crear estructura de directorios:**

```bash
mkdir -p ml_lib/[modulo]/{interfaces,models,services,handlers}
touch ml_lib/[modulo]/__init__.py
touch ml_lib/[modulo]/{interfaces,models,services,handlers}/__init__.py
```

2. **Definir interfaces en `interfaces/`:**

```python
from abc import ABC, abstractmethod

class MyInterface(ABC):
    @abstractmethod
    def method(self) -> None:
        pass
```

3. **Crear modelos en `models/`:**

```python
from dataclasses import dataclass

@dataclass
class MyModel:
    field: str
    value: float
```

4. **Implementar servicios en `services/`:**

```python
class MyService:
    def __init__(self, dependency: SomeDependency):
        self.dependency = dependency

    def perform_operation(self) -> Result:
        # Implementation
        pass
```

5. **Añadir handlers en `handlers/`:**

```python
class MyHandler:
    def handle_error(self, error: Exception) -> None:
        # Handle error
        pass
```

6. **Exportar API en `__init__.py`:**

```python
from .interfaces import MyInterface
from .models import MyModel
from .services import MyService
from .handlers import MyHandler

__all__ = [
    "MyInterface",
    "MyModel",
    "MyService",
    "MyHandler",
]
```

### Verificar Estructura

```bash
# Ejecutar validación de estructura
python3 scripts/check_module_structure.py
```

## Roadmap y Backlog

El backlog está organizado en User Stories ubicadas en `docs/backlog/`. **Ver `docs/backlog/README.md` para el índice completo.**

### ⚡ PRIORIDAD CRÍTICA: Épica 0 - Code Quality Foundation

**DEBE COMPLETARSE ANTES de continuar con nuevos módulos**

**00_code_quality/** - Fundamentos de calidad de código (66 horas estimadas)

- **US 0.1**: Refactorización a Clases con Tipado Fuerte (28h)

  - Eliminar uso excesivo de diccionarios
  - Convertir strings mágicos en Enums
  - Crear clases dataclass bien tipadas

- **US 0.2**: Seguridad de Tipos Completa (20h)

  - Eliminar uso innecesario de `Any`
  - Configurar mypy en modo strict
  - Implementar Generics correctamente
  - Usar numpy.typing

- **US 0.3**: Validación y Robustez (18h)
  - Crear jerarquía de excepciones
  - Validación en todas las dataclasses
  - Decoradores de validación
  - ArrayValidator completo

### Épicas Implementadas

1. **01_core_infrastructure/** - Infraestructura base ✅ (Implementado)

   - US 1.1: Interfaces consistentes
   - US 1.2: Validación de entradas
   - US 1.3: Manejo de errores y logging
   - US 1.4: Modelo base

2. **02_linear_algebra/** - Álgebra lineal ✅ (Implementado)
   - US 2.1: Operaciones optimizadas
   - US 2.2: Descomposiciones matriciales
   - US 2.3: Matrices dispersas

### Épicas en Desarrollo

3. **03_automatic_differentiation/** - Diferenciación automática 🚧 (En desarrollo)

   - US 3.1: Grafo computacional
   - US 3.2: Backpropagation

4. **04_numerical_optimization/** - Optimización numérica 🚧 (En desarrollo)
   - US 4.1: Optimizadores de primer orden
   - US 4.2: Optimizadores de segundo orden
   - US 4.3: Schedulers de learning rate

### Épicas Planificadas

5. **05_kernel_methods/** - Métodos de kernel 📋 (Planificado)
6. **06_probabilistic_models/** - Modelos probabilísticos 📋 (Planificado)
7. **07_deep_learning/** - Deep learning 📋 (Planificado)
8. **08_ensemble_learning/** - Ensemble learning 📋 (Planificado)
9. **09_feature_engineering/** - Feature engineering 📋 (Planificado)
10. **10_data_handling_at_scale/** - Procesamiento a escala 📋 (Planificado)
11. **11_uncertainty_quantification/** - Cuantificación de incertidumbre 📋 (Planificado)
12. **12_time_series/** - Series temporales 📋 (Planificado)
13. **13_reinforcement_learning/** - RL 📋 (Planificado)

## Comandos Útiles

### Desarrollo

```bash
# Crear entorno virtual
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
uv pip install -e .

# Instalar dependencias de desarrollo
uv pip install -e ".[dev]"

# Ejecutar tests
pytest

# Type checking
mypy ml_lib

# Formateo de código
black ml_lib
isort ml_lib

# Linting
flake8 ml_lib

# Validar estructura
python3 scripts/check_module_structure.py
```

### Demostración

```bash
# Ejecutar aplicación de demostración
cd .
PYTHONPATH=. python3 ecoml_analyzer/main.py
```

## Convenciones de Código

### Naming Conventions

- **Módulos**: `snake_case` (ej. `linear_algebra`)
- **Clases**: `PascalCase` (ej. `EstimatorInterface`)
- **Funciones/métodos**: `snake_case` (ej. `fit_transform`)
- **Constantes**: `UPPER_SNAKE_CASE` (ej. `MAX_ITERATIONS`)
- **Variables privadas**: `_leading_underscore` (ej. `_internal_state`)

### Type Hints

- **Siempre usar type hints** en signatures de funciones
- Usar `Generic` y `TypeVar` para componentes reutilizables
- Usar `Protocol` para duck typing cuando sea apropiado
- Usar `Union` y `Optional` explícitamente

### Docstrings

```python
def function(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When conditions are not met
    """
    pass
```

### Imports

```python
# Standard library
import logging
from typing import Any, Dict

# Third-party
import numpy as np
import pandas as pd

# Local
from ml_lib.core.interfaces import EstimatorInterface
from ml_lib.core.services import ValidationService
```

## Testing

### Estructura de Tests

```
tests/
├── unit/                  # Tests unitarios
│   ├── core/
│   ├── linalg/
│   └── ...
├── integration/           # Tests de integración
│   └── ...
└── performance/           # Tests de rendimiento
    └── ...
```

### Ejemplo de Test

```python
import pytest
from ml_lib.core.services import ValidationService

def test_validation_service_shape():
    service = ValidationService(logger=mock_logger)
    X = np.array([[1, 2], [3, 4]])

    # Should not raise
    service.validate_input_shape(X, 2)

    # Should raise
    with pytest.raises(ValueError):
        service.validate_input_shape(X, 1)
```

## Referencias y Documentación

### Documentos Clave

- **`claude.md`** (este archivo): Contexto completo del proyecto para Claude Code
- **`docs/CODE_QUALITY_GUIDELINES.md`**: Guía de calidad de código (clases vs dicts, enums)
- **`docs/backlog/README.md`**: Índice completo del product backlog
- **`docs/backlog/00_code_quality/`**: User stories de calidad de código (PRIORIDAD)
- **`docs/ml_library_structure.py`**: Estructura completa del proyecto con ejemplos
- **`docs/ml_advanced_course.md`**: Curso avanzado de ML con teoría
- **`README.md`**: Documentación de uso general
- **`ecoml_analyzer/README.md`**: Documentación de la aplicación de demostración

### Referencias Externas

- Python Type Hints: https://docs.python.org/3/library/typing.html
- NumPy Documentation: https://numpy.org/doc/
- SciPy Documentation: https://docs.scipy.org/doc/
- Design Patterns: https://refactoring.guru/design-patterns

## Contribuciones

### Proceso de Contribución

1. Revisar el backlog en `docs/backlog/`
2. Seleccionar una User Story
3. Crear módulo siguiendo la estructura estándar
4. Implementar interfaces, modelos, servicios y handlers
5. Añadir tests
6. Validar estructura con `scripts/check_module_structure.py`
7. Ejecutar type checking, linting y tests
8. Actualizar documentación

### Checklist antes de Commit

- [ ] Estructura de módulo validada
- [ ] Type hints completos
- [ ] Docstrings añadidos
- [ ] Tests unitarios pasando
- [ ] mypy sin errores
- [ ] black y isort aplicados
- [ ] flake8 sin warnings
- [ ] Documentación actualizada

## Estado Actual del Proyecto

### ✅ Completado

- Core infrastructure con interfaces base
- Sistema de validación y logging
- Módulo de álgebra lineal completo
- Módulo de visualización general
- Script de validación de estructura
- Aplicación de demostración EcoML Analyzer

### 🚧 En Progreso

- Automatic differentiation
- Numerical optimization
- Testing completo de módulos existentes

### 📋 Pendiente

- Kernel methods
- Probabilistic models
- Neural networks
- Ensemble methods
- Feature engineering
- Data processing a escala
- Uncertainty quantification
- Time series
- Reinforcement learning
- Interpretability
- AutoML
- Fairness
- Deployment
- Plugin system
- Performance optimization

## Notas Importantes para Claude Code

### ⚡ PRIORIDAD MÁXIMA - Code Quality

**ANTES de trabajar en nuevos módulos o features:**

1. **LEER** `docs/CODE_QUALITY_GUIDELINES.md` completamente
2. **REVISAR** el backlog de calidad en `docs/backlog/00_code_quality/`
3. **APLICAR** los principios de calidad a TODO el código nuevo

### Reglas Críticas de Calidad

#### Clases sobre Diccionarios ❌ NO `Dict[str, Any]`

- ❌ **NUNCA** usar `Dict[str, Any]` en APIs públicas
- ✅ **SIEMPRE** crear dataclasses con tipos específicos
- 📖 **VER** `docs/CODE_QUALITY_GUIDELINES.md`

#### Enums sobre Strings ❌ NO strings mágicos

- ❌ **NUNCA** usar strings mágicos para opciones limitadas
- ✅ **SIEMPRE** crear Enums para opciones fijas
- 📖 **VER** `docs/backlog/00_code_quality/US_0.1_refactor_to_classes.md`

#### Validación Obligatoria ✅

- ✅ **SIEMPRE** añadir validación en `__post_init__`
- ✅ **CREAR** excepciones personalizadas descriptivas
- 📖 **VER** `docs/backlog/00_code_quality/US_0.3_validation_and_robustness.md`

#### Type Hints Completos ✅

- ❌ **NUNCA** usar `Any` sin justificación
- ✅ **USAR** `numpy.typing` para arrays
- 📖 **VER** `docs/backlog/00_code_quality/US_0.2_type_safety.md`

#### Interfaces Limpias - No Tuplas ni Dicts Confusos ❌

**Principio: El usuario NO debe adivinar qué retorna una función**

- ❌ **NUNCA** retornar `Tuple[np.ndarray, Dict[str, Any]]` en interfaces públicas
- ❌ **NUNCA** retornar tuplas con >2 elementos sin documentación clara
- ✅ **SIEMPRE** crear dataclasses de resultado para operaciones complejas
- ✅ **USAR** nombres semánticos en lugar de índices numéricos
- 📖 **VER** `docs/architecture/INTERFACE_IMPROVEMENTS.md`

**Ejemplos:**

```python
# ❌ MAL - Usuario debe adivinar
def predict_with_metadata(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    predictions = model.predict(X)
    metadata = {"confidence": 0.95, "time_ms": 123}
    return predictions, metadata

# Usuario debe recordar orden y adivinar claves del dict
preds, meta = model.predict_with_metadata(X)
confidence = meta["confidence"]  # ¿Existe esta clave?

# ✅ BIEN - Claridad total
@dataclass
class PredictionResult:
    """Resultado de predicción con metadatos."""
    predictions: np.ndarray
    confidence_scores: np.ndarray
    execution_time_ms: float
    feature_importances: Optional[np.ndarray] = None

def predict_detailed(X: np.ndarray) -> PredictionResult:
    result = PredictionResult(
        predictions=model.predict(X),
        confidence_scores=confidence,
        execution_time_ms=123.5
    )
    return result

# Usuario tiene autocompletado y claridad
result = model.predict_detailed(X)
result.predictions  # ✅ IDE muestra qué está disponible
result.confidence_scores  # ✅ Tipos claros
```

**Excepciones legítimas para tuplas**:

- Pares matemáticos universales (como meshgrid → `(X, Y)`)
- Descomposiciones estándar (QR → `(Q, R)`, pero mejor usar dataclass)
- Cuando hay SOLO 2 elementos con semántica obvia

### Checklist Antes de Cualquier Commit

- [ ] ✅ Estructura validada: `python3 scripts/check_module_structure.py`
- [ ] ❌ No hay `Dict[str, Any]` en APIs públicas
- [ ] ❌ No hay strings mágicos (usar Enums)
- [ ] ✅ Validación en todas las dataclasses
- [ ] ✅ Type hints completos
- [ ] ✅ Tests pasando (unitarios + validación)
- [ ] ✅ mypy sin errores
- [ ] ✅ black + isort aplicados

---

**Última actualización:** 2025-10-09
**Versión:** 0.1.0
**Licencia:** MIT
