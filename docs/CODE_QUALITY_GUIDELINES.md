# Guía de Calidad de Código - ML Library

## Principios Fundamentales

### 1. **Clases sobre Diccionarios**

❌ **MAL - Uso excesivo de diccionarios:**
```python
# Diccionario con claves mágicas - difícil de mantener
def process_config(config: dict) -> dict:
    result = {
        "status": "ok",
        "value": config["input"] * 2,
        "metadata": {
            "timestamp": "2024-01-01",
            "version": "1.0"
        }
    }
    return result
```

✅ **BIEN - Clases con tipado:**
```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class Status(Enum):
    OK = "ok"
    ERROR = "error"
    PENDING = "pending"

@dataclass
class Metadata:
    timestamp: datetime
    version: str

@dataclass
class ProcessResult:
    status: Status
    value: float
    metadata: Metadata

def process_config(config: Config) -> ProcessResult:
    return ProcessResult(
        status=Status.OK,
        value=config.input_value * 2,
        metadata=Metadata(
            timestamp=datetime.now(),
            version="1.0"
        )
    )
```

### 2. **Uso Apropiado de Enums**

❌ **MAL - Strings mágicos:**
```python
def train_model(model_type: str, optimizer: str):
    if model_type == "linear":  # Propenso a errores de tipeo
        pass
    elif model_type == "neural":
        pass

    if optimizer == "sgd":  # Sin autocompletado
        pass
```

✅ **BIEN - Enums fuertemente tipados:**
```python
from enum import Enum, auto

class ModelType(Enum):
    LINEAR = auto()
    NEURAL = auto()
    TREE = auto()
    ENSEMBLE = auto()

class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    LBFGS = "lbfgs"

def train_model(model_type: ModelType, optimizer: OptimizerType):
    if model_type == ModelType.LINEAR:
        pass
    elif model_type == ModelType.NEURAL:
        pass

    if optimizer == OptimizerType.SGD:
        pass
```

### 3. **Cuándo Usar Diccionarios (Casos Permitidos)**

✅ **USO VÁLIDO - Datos dinámicos/JSON:**
```python
@dataclass
class APIResponse:
    status_code: int
    headers: dict[str, str]  # Headers son inherentemente dinámicos
    body: dict[str, Any]  # JSON response puede tener estructura variable
```

✅ **USO VÁLIDO - Mapeos de configuración:**
```python
@dataclass
class ModelRegistry:
    models: dict[str, type[EstimatorInterface]]  # Registro dinámico de clases

    def register(self, name: str, model_class: type[EstimatorInterface]):
        self.models[name] = model_class
```

✅ **USO VÁLIDO - Parámetros kwargs:**
```python
def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> "Estimator":
    # fit_params puede contener opciones variables según el contexto
    pass
```

## Patrones de Diseño Mejorados

### Patrón 1: Estados del Modelo con Enum

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class ModelState(Enum):
    """Estado del ciclo de vida del modelo."""
    INITIALIZED = auto()
    TRAINING = auto()
    FITTED = auto()
    VALIDATING = auto()
    READY = auto()
    FAILED = auto()

@dataclass
class ModelStatus:
    """Estado completo del modelo."""
    state: ModelState
    error_message: Optional[str] = None
    training_iteration: int = 0
    last_loss: Optional[float] = None

    def is_ready(self) -> bool:
        return self.state == ModelState.READY

    def has_failed(self) -> bool:
        return self.state == ModelState.FAILED
```

### Patrón 2: Configuración Tipada

```python
from dataclasses import dataclass, field
from enum import Enum

class ActivationFunction(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"

class InitializationStrategy(Enum):
    XAVIER = "xavier"
    HE = "he"
    NORMAL = "normal"
    UNIFORM = "uniform"

@dataclass
class LayerConfig:
    """Configuración de capa neural."""
    units: int
    activation: ActivationFunction
    use_bias: bool = True
    kernel_initializer: InitializationStrategy = InitializationStrategy.XAVIER
    bias_initializer: InitializationStrategy = InitializationStrategy.NORMAL
    dropout_rate: float = 0.0

    def __post_init__(self):
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.units <= 0:
            raise ValueError("units must be positive")

@dataclass
class NetworkConfig:
    """Configuración de red neural completa."""
    layers: list[LayerConfig]
    optimizer: OptimizerType
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

    def validate(self) -> None:
        """Valida la configuración completa."""
        if not self.layers:
            raise ValueError("Network must have at least one layer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
```

### Patrón 3: Resultados Estructurados

```python
from dataclasses import dataclass
from typing import Generic, TypeVar
from enum import Enum

T = TypeVar('T')

class ValidationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"

@dataclass
class ValidationResult(Generic[T]):
    """Resultado de validación genérico."""
    status: ValidationStatus
    data: T
    message: str
    warnings: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return self.status != ValidationStatus.FAILED

@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento."""
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    epoch: int

@dataclass
class TrainingResult:
    """Resultado completo del entrenamiento."""
    final_metrics: TrainingMetrics
    history: list[TrainingMetrics]
    total_epochs: int
    convergence_achieved: bool
    training_time_seconds: float
    best_epoch: int
```

### Patrón 4: Tipos de Error Específicos con Enum

```python
from enum import Enum

class ErrorCategory(Enum):
    """Categorías de errores."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    CONFIGURATION = "configuration"
    DATA = "data"
    CONVERGENCE = "convergence"

class ErrorSeverity(Enum):
    """Severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MLError:
    """Error estructurado de ML."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: dict[str, Any]  # Uso válido: información de contexto dinámica
    timestamp: datetime = field(default_factory=datetime.now)

    def is_critical(self) -> bool:
        return self.severity == ErrorSeverity.CRITICAL
```

## Mejoras Específicas por Módulo

### Core Module

```python
# core/models/training_mode.py
from enum import Enum

class TrainingMode(Enum):
    """Modo de entrenamiento."""
    BATCH = "batch"
    ONLINE = "online"
    MINI_BATCH = "mini_batch"

class ValidationStrategy(Enum):
    """Estrategia de validación."""
    HOLDOUT = "holdout"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"

# core/models/base_model.py
@dataclass
class BaseModel:
    """Modelo base mejorado."""
    name: str
    version: str
    state: ModelState = ModelState.INITIALIZED
    status: ModelStatus = field(default_factory=lambda: ModelStatus(state=ModelState.INITIALIZED))
    metadata: Metadata = field(default_factory=Metadata)  # Clase, no dict

    # ❌ ELIMINADO: metadata: Dict[str, Any]
    # ✅ NUEVO: metadata tipado con clase
```

### Linalg Module

```python
# linalg/models/precision.py
from enum import Enum

class MatrixPrecision(Enum):
    """Precisión de matriz."""
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    FLOAT16 = "float16"

class MemoryLayout(Enum):
    """Layout de memoria."""
    ROW_MAJOR = "C"
    COLUMN_MAJOR = "F"

class DecompositionMethod(Enum):
    """Métodos de descomposición."""
    LU = "lu"
    QR = "qr"
    SVD = "svd"
    CHOLESKY = "cholesky"
    EIGEN = "eigen"

@dataclass
class MatrixOperationConfig:
    """Configuración de operación matricial."""
    precision: MatrixPrecision = MatrixPrecision.FLOAT64
    layout: MemoryLayout = MemoryLayout.ROW_MAJOR
    use_blas: bool = True
    parallel: bool = True
    num_threads: int = -1  # -1 = auto
```

### Optimization Module

```python
# optimization/models/optimizer_config.py
from enum import Enum

class OptimizerType(Enum):
    """Tipos de optimizadores."""
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
    """Tipos de schedulers."""
    CONSTANT = "constant"
    STEP_DECAY = "step_decay"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"

class ConvergenceCriterion(Enum):
    """Criterios de convergencia."""
    GRADIENT_NORM = "gradient_norm"
    LOSS_CHANGE = "loss_change"
    PARAMETER_CHANGE = "parameter_change"
    MAX_ITERATIONS = "max_iterations"

@dataclass
class OptimizerConfig:
    """Configuración de optimizador."""
    optimizer_type: OptimizerType
    learning_rate: float
    momentum: float = 0.0
    weight_decay: float = 0.0
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999

@dataclass
class ConvergenceConfig:
    """Configuración de convergencia."""
    criterion: ConvergenceCriterion
    tolerance: float = 1e-6
    max_iterations: int = 1000
    patience: int = 10  # Para early stopping
```

### Neural Networks Module

```python
# neural/models/network_config.py
from enum import Enum

class ActivationType(Enum):
    """Tipos de activación."""
    LINEAR = "linear"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    SWISH = "swish"

class LossType(Enum):
    """Tipos de función de pérdida."""
    MSE = "mse"
    MAE = "mae"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    CATEGORICAL_CROSS_ENTROPY = "categorical_cross_entropy"
    HUBER = "huber"
    HINGE = "hinge"

class RegularizationType(Enum):
    """Tipos de regularización."""
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
```

### Visualization Module

```python
# visualization/models/plot_types.py
from enum import Enum

class PlotType(Enum):
    """Tipos de gráficos."""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    SURFACE = "surface"

class ColorScheme(Enum):
    """Esquemas de color."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"

class PlotStyle(Enum):
    """Estilos de gráfico."""
    DEFAULT = "default"
    SEABORN = "seaborn"
    GGPLOT = "ggplot"
    DARK_BACKGROUND = "dark_background"
    CLASSIC = "classic"

@dataclass
class PlotConfig:
    """Configuración mejorada de gráfico."""
    plot_type: PlotType
    title: str
    xlabel: str = ""
    ylabel: str = ""
    style: PlotStyle = PlotStyle.DEFAULT
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    figsize: tuple[int, int] = (10, 6)
    grid: bool = True
    legend: bool = True

    # ❌ ELIMINADO: additional_params: Dict[str, Any]
    # Si necesitas más configuración, añade campos específicos
```

## Checklist de Calidad

### Antes de Crear una Clase:

- [ ] ¿Estoy usando `dict` donde debería usar una clase?
- [ ] ¿Hay strings mágicos que deberían ser Enums?
- [ ] ¿Los campos tienen tipos específicos o uso `Any` innecesariamente?
- [ ] ¿He definido validación en `__post_init__`?
- [ ] ¿La clase tiene métodos de conveniencia útiles?

### Antes de Usar un Dict:

- [ ] ¿Es realmente necesario el dict o puedo definir una clase?
- [ ] ¿Estoy parseando JSON/datos externos donde la estructura varía?
- [ ] ¿Es un mapeo genuinamente dinámico (como un registro)?
- [ ] ¿Puedo usar TypedDict al menos para documentar la estructura?

### Antes de Usar un String:

- [ ] ¿Este string representa un conjunto limitado de opciones?
- [ ] ¿Debería ser un Enum para evitar errores de tipeo?
- [ ] ¿Necesito autocompletado y type checking?
- [ ] ¿Este valor se compara o usa en lógica condicional?

## Refactoring de Código Existente

### Ejemplo: Refactoring de BaseModel

**ANTES:**
```python
@dataclass
class BaseModel:
    name: str
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)  # ❌
    is_fitted: bool = False  # ❌ Debería ser enum
```

**DESPUÉS:**
```python
from enum import Enum

class ModelState(Enum):
    INITIALIZED = auto()
    TRAINING = auto()
    FITTED = auto()
    READY = auto()

@dataclass
class ModelMetadata:
    """Metadata estructurado."""
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    tags: list[str] = field(default_factory=list)

@dataclass
class BaseModel:
    name: str
    version: str
    metadata: ModelMetadata  # ✅ Clase tipada
    state: ModelState = ModelState.INITIALIZED  # ✅ Enum
```

## Prioridades de Refactoring

1. **Alta Prioridad:**
   - Reemplazar `Dict[str, Any]` en configuraciones
   - Convertir strings mágicos en Enums
   - Añadir tipos específicos donde hay `Any`

2. **Media Prioridad:**
   - Crear clases para estructuras de datos complejas
   - Añadir validación en `__post_init__`
   - Documentar clases y métodos

3. **Baja Prioridad:**
   - Optimizar imports
   - Refactorizar código legacy que funciona
   - Añadir más métodos de conveniencia

## Recursos

- Python Dataclasses: https://docs.python.org/3/library/dataclasses.html
- Python Enum: https://docs.python.org/3/library/enum.html
- Type Hints: https://docs.python.org/3/library/typing.html
- PEP 589 (TypedDict): https://peps.python.org/pep-0589/

---

**Última actualización:** 2025-10-09
