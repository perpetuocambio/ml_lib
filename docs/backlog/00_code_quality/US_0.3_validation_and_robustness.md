# User Story 0.3: Validación y Robustez

**Como usuario de la biblioteca,** quiero que todos los componentes validen sus entradas y proporcionen mensajes de error claros para detectar problemas temprano y facilitar el debugging.

## Objetivos

1. Validación completa en todas las clases
2. Excepciones personalizadas y descriptivas
3. Mensajes de error informativos
4. Defensive programming

## Tareas

### Task 0.3.1: Crear Jerarquía de Excepciones

- **Descripción:** Definir excepciones personalizadas para diferentes tipos de errores
- **Entregable:** Módulo de excepciones
- **Prioridad:** Alta
- **Estimación:** 3 horas

```python
# core/exceptions.py
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MLLibException(Exception):
    """Excepción base para ml_lib."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.severity = severity
        super().__init__(message)

class ValidationError(MLLibException):
    """Error de validación de datos."""
    pass

class ShapeError(ValidationError):
    """Error de forma de array."""
    def __init__(self, expected: tuple[int, ...], actual: tuple[int, ...]):
        self.expected = expected
        self.actual = actual
        message = f"Expected shape {expected}, got {actual}"
        super().__init__(message)

class NotFittedError(MLLibException):
    """Modelo no ha sido entrenado."""
    pass

class ConvergenceError(MLLibException):
    """Error de convergencia en optimización."""
    pass

class ConfigurationError(MLLibException):
    """Error en configuración."""
    pass
```

**Checklist:**
- [ ] Definir jerarquía de excepciones
- [ ] Documentar cada excepción
- [ ] Crear ejemplos de uso
- [ ] Añadir tests

### Task 0.3.2: Validación en __post_init__

- **Descripción:** Añadir validación automática en todas las dataclasses
- **Entregable:** Todas las dataclasses con validación
- **Prioridad:** Alta
- **Estimación:** 4 horas

```python
from dataclasses import dataclass
from ml_lib.core.exceptions import ValidationError

@dataclass
class LayerConfig:
    units: int
    dropout_rate: float = 0.0

    def __post_init__(self):
        """Valida configuración al crear instancia."""
        self._validate_units()
        self._validate_dropout()

    def _validate_units(self):
        if self.units <= 0:
            raise ValidationError(
                f"units must be positive, got {self.units}"
            )

    def _validate_dropout(self):
        if not 0 <= self.dropout_rate <= 1:
            raise ValidationError(
                f"dropout_rate must be in [0, 1], got {self.dropout_rate}"
            )
```

**Checklist:**
- [ ] Auditar todas las dataclasses
- [ ] Añadir `__post_init__` con validación
- [ ] Separar validaciones en métodos privados
- [ ] Añadir tests de validación

### Task 0.3.3: Decorador de Validación

- **Descripción:** Crear decorador para validación de métodos
- **Entregable:** Decorador reutilizable
- **Prioridad:** Media
- **Estimación:** 3 horas

```python
# core/decorators.py
from functools import wraps
from typing import Callable, TypeVar
import numpy as np

T = TypeVar('T')

def validate_fitted(func: Callable[..., T]) -> Callable[..., T]:
    """Decorador que verifica que el modelo esté entrenado."""
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> T:
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise NotFittedError(
                f"{self.__class__.__name__} must be fitted before calling {func.__name__}"
            )
        return func(self, *args, **kwargs)
    return wrapper

def validate_input_shape(expected_dims: int):
    """Decorador que valida dimensiones de entrada."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, X: np.ndarray, *args, **kwargs) -> T:
            if X.ndim != expected_dims:
                raise ShapeError(
                    expected=(expected_dims,),
                    actual=(X.ndim,)
                )
            return func(self, X, *args, **kwargs)
        return wrapper
    return decorator

# Uso:
class Estimator:
    @validate_fitted
    @validate_input_shape(2)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
```

**Checklist:**
- [ ] Crear decoradores comunes
- [ ] Documentar uso
- [ ] Aplicar en código existente
- [ ] Añadir tests

### Task 0.3.4: Mensajes de Error Informativos

- **Descripción:** Mejorar mensajes de error con contexto útil
- **Entregable:** Template de mensajes de error
- **Prioridad:** Media
- **Estimación:** 2 horas

**Principios:**
- Incluir valores esperados vs actuales
- Sugerir soluciones
- Incluir contexto relevante
- Ser concisos pero informativos

```python
# ❌ MAL
raise ValueError("Invalid value")

# ✅ BIEN
raise ValidationError(
    f"learning_rate must be positive, got {learning_rate}. "
    f"Try values like 0.001, 0.01, or 0.1."
)

# ✅ MUY BIEN con contexto
raise ShapeError(
    expected=(None, self.n_features_),
    actual=X.shape,
    message=(
        f"Input data has {X.shape[1]} features, but estimator "
        f"was fitted with {self.n_features_} features. "
        f"Ensure all input data has the same number of features."
    )
)
```

**Checklist:**
- [ ] Auditar mensajes de error existentes
- [ ] Crear templates
- [ ] Actualizar excepciones
- [ ] Documentar mejores prácticas

### Task 0.3.5: Validación de Arrays NumPy

- **Descripción:** Crear funciones de validación para arrays
- **Entregable:** Módulo de validación de arrays
- **Prioridad:** Alta
- **Estimación:** 3 horas

```python
# core/validation.py
import numpy as np
from ml_lib.core.exceptions import ValidationError

class ArrayValidator:
    """Validador de arrays NumPy."""

    @staticmethod
    def check_array(
        X: np.ndarray,
        *,
        dtype: str = "numeric",
        ndim: int | None = None,
        min_samples: int = 1,
        min_features: int = 1,
        allow_nan: bool = False,
        allow_inf: bool = False,
        ensure_finite: bool = True,
        name: str = "X"
    ) -> np.ndarray:
        """
        Valida array de entrada.

        Args:
            X: Array a validar
            dtype: Tipo de dato esperado
            ndim: Número de dimensiones esperado
            min_samples: Número mínimo de muestras
            min_features: Número mínimo de features
            allow_nan: Permitir NaN
            allow_inf: Permitir infinitos
            ensure_finite: Asegurar valores finitos
            name: Nombre del array para mensajes de error

        Returns:
            Array validado

        Raises:
            ValidationError: Si la validación falla
        """
        # Validar tipo
        if not isinstance(X, np.ndarray):
            raise ValidationError(
                f"{name} must be numpy array, got {type(X)}"
            )

        # Validar dimensiones
        if ndim is not None and X.ndim != ndim:
            raise ShapeError(
                expected=(ndim,),
                actual=(X.ndim,),
                message=f"{name} must be {ndim}D array"
            )

        # Validar tamaño
        if X.shape[0] < min_samples:
            raise ValidationError(
                f"{name} must have at least {min_samples} samples, "
                f"got {X.shape[0]}"
            )

        if X.ndim >= 2 and X.shape[1] < min_features:
            raise ValidationError(
                f"{name} must have at least {min_features} features, "
                f"got {X.shape[1]}"
            )

        # Validar valores
        if ensure_finite:
            if not allow_nan and np.any(np.isnan(X)):
                raise ValidationError(f"{name} contains NaN values")
            if not allow_inf and np.any(np.isinf(X)):
                raise ValidationError(f"{name} contains infinite values")

        return X
```

**Checklist:**
- [ ] Implementar `ArrayValidator`
- [ ] Añadir todas las validaciones comunes
- [ ] Integrar en estimadores
- [ ] Añadir tests exhaustivos

### Task 0.3.6: Contract Testing

- **Descripción:** Implementar tests de contrato para interfaces
- **Entregable:** Suite de contract tests
- **Prioridad:** Media
- **Estimación:** 3 horas

```python
# tests/contracts/test_estimator_contract.py
import pytest
from ml_lib.core.interfaces import EstimatorInterface

def test_estimator_contract(estimator_class):
    """Test que verifica que un estimador cumple el contrato."""
    # Debe poder instanciarse
    estimator = estimator_class()

    # Debe tener métodos requeridos
    assert hasattr(estimator, 'fit')
    assert hasattr(estimator, 'predict')
    assert hasattr(estimator, 'get_params')
    assert hasattr(estimator, 'set_params')

    # Debe implementar EstimatorInterface
    assert isinstance(estimator, EstimatorInterface)

    # fit debe retornar self
    X, y = np.random.randn(10, 5), np.random.randn(10)
    result = estimator.fit(X, y)
    assert result is estimator

    # predict debe funcionar después de fit
    predictions = estimator.predict(X)
    assert predictions.shape[0] == X.shape[0]
```

**Checklist:**
- [ ] Crear contract tests para interfaces
- [ ] Aplicar a todos los estimadores
- [ ] Documentar requisitos
- [ ] Integrar en CI/CD

## Criterios de Aceptación

- [ ] Todas las clases tienen validación
- [ ] Excepciones personalizadas definidas
- [ ] Mensajes de error informativos
- [ ] Decoradores de validación usados
- [ ] ArrayValidator integrado
- [ ] Contract tests pasando

## Estimación Total

**18 horas** (~2 días de desarrollo)
