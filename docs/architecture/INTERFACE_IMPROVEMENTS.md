# Mejoras de Interfaces - Eliminación de Dict[str, Any]

## Problema

Actualmente nuestras interfaces tienen algunos puntos donde retornan tipos genéricos poco claros:

### 1. `get_params()` retorna `dict[str, Any]`

**Ubicaciones**:
- `EstimatorInterface.get_params()` → `dict[str, Any]`
- `OptimizerInterface.get_params()` → `dict[str, Any]`

**Problema para el usuario**:
```python
# Usuario no sabe qué claves están disponibles
params = model.get_params()
lr = params["learning_rate"]  # ¿Existe esta clave?
# ❌ Sin autocompletado
# ❌ Sin validación de tipos
# ❌ Sin documentación inline
```

### 2. Tuplas en retornos de servicios

**Ubicaciones encontradas**:
- Descomposiciones matriciales (QR, LU, SVD) → `Tuple[np.ndarray, ...]`
- Verificaciones con mensajes → `Tuple[bool, str]`

## Solución Propuesta

### A. Sistema de Configuración Tipada

Crear clases de configuración específicas para cada estimador/optimizador:

```python
from dataclasses import dataclass
from typing import Protocol, TypeVar

@dataclass
class EstimatorConfig(Protocol):
    """Protocolo para configuraciones de estimadores."""
    pass

# Ejemplo: Configuración de un optimizador SGD
@dataclass
class SGDConfig:
    """Configuración para SGD fuertemente tipada."""
    learning_rate: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

# Interfaz mejorada
TConfig = TypeVar("TConfig", bound=EstimatorConfig)

class EstimatorInterface(ABC, Generic[X, Y, TConfig]):
    """Interface con configuración tipada."""

    @abstractmethod
    def get_config(self) -> TConfig:
        """Obtiene la configuración del estimador (tipada)."""
        pass

    @abstractmethod
    def set_config(self, config: TConfig) -> "EstimatorInterface[X, Y, TConfig]":
        """Establece la configuración desde un objeto tipado."""
        pass

    # Mantener para compatibilidad sklearn
    def get_params(self) -> dict[str, Any]:
        """Retorna parámetros como dict (sklearn compatibility)."""
        config = self.get_config()
        return asdict(config)

    def set_params(self, **params) -> "EstimatorInterface[X, Y, TConfig]":
        """Establece parámetros desde kwargs (sklearn compatibility)."""
        config_dict = asdict(self.get_config())
        config_dict.update(params)
        config = self._config_class(**config_dict)
        return self.set_config(config)
```

**Beneficios**:
```python
# Usuario tiene experiencia mejorada
config = sgd_optimizer.get_config()  # Retorna SGDConfig
config.learning_rate  # ✅ Autocompletado del IDE
config.momentum = 1.5  # ✅ Validación en __post_init__

# Aún compatible con sklearn
params_dict = sgd_optimizer.get_params()  # Retorna dict si necesario
```

### B. Clases de Resultado para Operaciones Complejas

Para operaciones que retornan múltiples valores relacionados, usar dataclasses en lugar de tuplas:

#### Caso 1: Descomposiciones matriciales

```python
@dataclass
class QRDecomposition:
    """Resultado de la descomposición QR."""
    Q: np.ndarray  # Matriz ortogonal
    R: np.ndarray  # Matriz triangular superior

    def __post_init__(self):
        """Validación de resultados."""
        if self.Q.shape[1] != self.R.shape[0]:
            raise ValueError("Dimensiones incompatibles Q y R")

@dataclass
class LUDecomposition:
    """Resultado de la descomposición LU."""
    L: np.ndarray  # Lower triangular
    U: np.ndarray  # Upper triangular
    P: np.ndarray  # Permutation matrix

@dataclass
class SVDDecomposition:
    """Resultado de la descomposición SVD."""
    U: np.ndarray  # Left singular vectors
    s: np.ndarray  # Singular values
    Vt: np.ndarray  # Right singular vectors (transposed)

# Uso
result = linalg_service.qr_decomposition(A)
result.Q  # ✅ Claro qué es cada componente
result.R  # ✅ Autocompletado
```

#### Caso 2: Verificaciones con contexto

```python
@dataclass
class VerificationResult:
    """Resultado de una verificación con contexto."""
    is_valid: bool
    message: str
    details: Optional[dict[str, Any]] = None
    severity: ErrorSeverity = ErrorSeverity.LOW

class ConvergenceResult(VerificationResult):
    """Resultado específico de verificación de convergencia."""
    iterations: int = 0
    final_loss: Optional[float] = None

# Uso
result = optimizer.check_convergence(history)
if not result.is_valid:
    logger.warning(result.message, extra={"severity": result.severity})
```

### C. Excepciones: Usos Legítimos de Tuplas

No todas las tuplas son malas. Las tuplas son apropiadas para:

1. **Retornos matemáticos estándar**:
```python
# ✅ ACEPTABLE - Convención matemática universal
def cartesian_product(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna meshgrid (X, Y)."""
    return np.meshgrid(x, y)
```

2. **Pares coordenada muy simples**:
```python
# ✅ ACEPTABLE - Par simple sin semántica compleja
def get_shape(matrix: np.ndarray) -> Tuple[int, int]:
    """Retorna (filas, columnas)."""
    return matrix.shape
```

3. **Cuando el orden es la única semántica**:
```python
# ✅ ACEPTABLE - Lista ordenada homogénea
def split_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (train, test)."""
    ...
```

**Criterio**: Si la tupla tiene más de 2 elementos O los elementos tienen semántica compleja → usar dataclass.

## Plan de Implementación

### Fase 1: Crear modelos de configuración (4 horas)
- [ ] `ml_lib/core/models/configs.py`
  - [ ] `EstimatorConfig` (protocolo base)
  - [ ] `OptimizerConfig` (protocolo base)
- [ ] Configs específicos por módulo
  - [ ] `optimization/models/configs.py` (SGDConfig, AdamConfig, etc.)

### Fase 2: Crear clases de resultado (3 horas)
- [ ] `ml_lib/linalg/models/results.py`
  - [ ] QRDecomposition, LUDecomposition, SVDDecomposition, EigenDecomposition
- [ ] `ml_lib/optimization_numérica_avanzada/models/results.py`
  - [ ] VerificationResult, ConvergenceResult

### Fase 3: Actualizar interfaces (2 horas)
- [ ] `EstimatorInterface`: añadir `get_config()`, `set_config()`
- [ ] `OptimizerInterface`: añadir `get_config()`, `set_config()`
- [ ] Mantener `get_params()` como wrapper para compatibilidad

### Fase 4: Migrar servicios (4 horas)
- [ ] Actualizar `linalg/services` para usar clases de resultado
- [ ] Actualizar `optimization_numérica_avanzada/handlers` para usar VerificationResult
- [ ] Mantener métodos legacy con @deprecated

### Fase 5: Tests y documentación (3 horas)
- [ ] Tests para validación de configs
- [ ] Tests para clases de resultado
- [ ] Actualizar docs con ejemplos
- [ ] Migration guide para usuarios

**Total estimado**: 16 horas

## Principios de Diseño

### 1. Claridad sobre compatibilidad
- Las interfaces deben ser **claras primero**
- Compatibilidad backwards se logra con wrappers, no comprometiendo la API principal

### 2. El usuario no debe adivinar
```python
# ❌ Usuario debe adivinar
result = model.predict_with_details(X)
predictions = result[0]  # ¿Qué es esto?
metadata = result[1]  # ¿Qué contiene?
confidence = result[1]["confidence"]  # ¿Existe esta clave?

# ✅ Usuario ve inmediatamente
result = model.predict_detailed(X)
predictions = result.predictions
confidence = result.confidence_scores  # Autocompletado!
```

### 3. Fail fast con validación
Todas las dataclasses de configuración deben validar en `__post_init__`:
```python
@dataclass
class OptimizerConfig:
    learning_rate: float

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
```

### 4. Documentación inline
Las clases autodocumentan:
```python
@dataclass
class AdamConfig:
    """Configuración para el optimizador Adam.

    Attributes:
        learning_rate: Tasa de aprendizaje inicial (default: 0.001)
        beta1: Coeficiente para el primer momento (default: 0.9)
        beta2: Coeficiente para el segundo momento (default: 0.999)
        epsilon: Término para estabilidad numérica (default: 1e-8)
    """
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
```

## Métricas de Éxito

1. **Cero `Dict[str, Any]` en interfaces públicas** (excepto wrappers de compatibilidad)
2. **Cero tuplas con >2 elementos** en retornos de servicios
3. **100% de configs con validación** en `__post_init__`
4. **Autocompletado funcional** en todos los IDEs principales (VS Code, PyCharm)
5. **mypy strict pasa** sin errores en módulos refactorizados

## Referencias

- [PEP 589 – TypedDict](https://peps.python.org/pep-0589/) (no recomendado para nosotros - preferimos dataclasses)
- [PEP 544 – Protocols](https://peps.python.org/pep-0544/) (usado para EstimatorConfig)
- [dataclasses documentation](https://docs.python.org/3/library/dataclasses.html)
