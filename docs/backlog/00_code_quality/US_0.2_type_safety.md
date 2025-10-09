# User Story 0.2: Seguridad de Tipos Completa

**Como desarrollador de la biblioteca,** quiero tener type hints completos y correctos en todo el código para aprovechar al máximo el type checking y mejorar la experiencia de desarrollo.

## Contexto

Algunos módulos tienen type hints incompletos o usan `Any` donde podrían ser más específicos.

## Objetivos

1. Eliminar uso innecesario de `Any`
2. Añadir type hints faltantes
3. Usar Generics donde sea apropiado
4. Configurar mypy en modo strict

## Tareas

### Task 0.2.1: Configurar mypy Strict

- **Descripción:** Configurar mypy en modo strict para el proyecto
- **Entregable:** Archivo `mypy.ini` configurado
- **Prioridad:** Alta
- **Estimación:** 1 hora

```ini
# mypy.ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
check_untyped_defs = True
strict_equality = True

[mypy-tests.*]
disallow_untyped_defs = False
```

**Checklist:**
- [ ] Crear `mypy.ini`
- [ ] Configurar en CI/CD
- [ ] Documentar excepciones permitidas
- [ ] Ejecutar y corregir errores iniciales

### Task 0.2.2: Reemplazar Any por Tipos Específicos

- **Descripción:** Buscar y reemplazar todos los `Any` innecesarios
- **Entregable:** Código sin `Any` innecesarios
- **Prioridad:** Alta
- **Estimación:** 4 horas

**Patrones comunes:**

```python
# ANTES ❌
def process(data: Any) -> Any:
    pass

# DESPUÉS ✅
from typing import TypeVar
T = TypeVar('T')

def process(data: T) -> T:
    pass

# O más específico:
def process(data: np.ndarray) -> np.ndarray:
    pass
```

**Checklist:**
- [ ] Auditar usos de `Any`
- [ ] Reemplazar con tipos específicos
- [ ] Usar TypeVar donde sea necesario
- [ ] Usar Union para tipos múltiples
- [ ] Actualizar tests

### Task 0.2.3: Generics y TypeVars

- **Descripción:** Usar Generics correctamente en interfaces y clases
- **Entregable:** Interfaces genéricas bien tipadas
- **Prioridad:** Alta
- **Estimación:** 3 horas

```python
from typing import Generic, TypeVar, Protocol
import numpy as np

# Definir TypeVars apropiados
X_co = TypeVar('X_co', bound=np.ndarray, covariant=True)
Y_co = TypeVar('Y_co', bound=np.ndarray, covariant=True)
Model_T = TypeVar('Model_T', bound='BaseModel')

class EstimatorInterface(Generic[X_co, Y_co], Protocol):
    def fit(self, X: X_co, y: Y_co) -> "EstimatorInterface[X_co, Y_co]":
        ...

    def predict(self, X: X_co) -> Y_co:
        ...
```

**Checklist:**
- [ ] Identificar clases que deben ser genéricas
- [ ] Definir TypeVars apropiados
- [ ] Usar varianza correctamente (covariant/contravariant)
- [ ] Actualizar implementaciones
- [ ] Actualizar tests

### Task 0.2.4: Protocol vs ABC

- **Descripción:** Evaluar cuándo usar Protocol vs ABC
- **Entregable:** Guía de uso y refactorización donde corresponda
- **Prioridad:** Media
- **Estimación:** 3 horas

**Guía:**
- **ABC**: Cuando necesitas implementación compartida o herencia explícita
- **Protocol**: Para duck typing y structural subtyping

```python
# ABC - Para herencia explícita
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    def common_method(self):  # Implementación compartida
        return "common"

# Protocol - Para duck typing
from typing import Protocol

class Fittable(Protocol):
    def fit(self, X, y) -> None:
        ...
```

**Checklist:**
- [ ] Documentar cuándo usar cada uno
- [ ] Evaluar interfaces existentes
- [ ] Refactorizar donde sea apropiado
- [ ] Crear ejemplos

### Task 0.2.5: Type Narrowing y Type Guards

- **Descripción:** Implementar type guards para validación de tipos en runtime
- **Entregable:** Type guards útiles
- **Prioridad:** Media
- **Estimación:** 2 horas

```python
from typing import TypeGuard
import numpy as np

def is_2d_array(arr: np.ndarray) -> TypeGuard[np.ndarray]:
    """Type guard para arrays 2D."""
    return arr.ndim == 2

def process_data(data: np.ndarray) -> None:
    if is_2d_array(data):
        # Aquí mypy sabe que data es 2D
        rows, cols = data.shape
```

**Checklist:**
- [ ] Identificar casos donde type guards son útiles
- [ ] Implementar type guards
- [ ] Usar en validaciones
- [ ] Documentar

### Task 0.2.6: Overload para Firmas Múltiples

- **Descripción:** Usar `@overload` para funciones con múltiples firmas
- **Entregable:** Overloads documentados
- **Prioridad:** Media
- **Estimación:** 2 horas

```python
from typing import overload, Union

@overload
def predict(self, X: np.ndarray) -> np.ndarray: ...

@overload
def predict(self, X: list[float]) -> list[float]: ...

def predict(self, X: Union[np.ndarray, list[float]]) -> Union[np.ndarray, list[float]]:
    # Implementación real
    pass
```

**Checklist:**
- [ ] Identificar funciones con múltiples firmas
- [ ] Añadir overloads
- [ ] Documentar
- [ ] Verificar con mypy

### Task 0.2.7: Type Aliases

- **Descripción:** Definir type aliases para tipos complejos
- **Entregable:** Módulo con type aliases comunes
- **Prioridad:** Baja
- **Estimación:** 2 horas

```python
# ml_lib/core/types.py
from typing import TypeAlias
import numpy as np
import numpy.typing as npt

# Type aliases útiles
NDArrayFloat: TypeAlias = npt.NDArray[np.floating]
NDArrayInt: TypeAlias = npt.NDArray[np.integer]
Shape: TypeAlias = tuple[int, ...]
ArrayLike: TypeAlias = Union[list, tuple, np.ndarray]
```

**Checklist:**
- [ ] Crear módulo `core/types.py`
- [ ] Definir aliases comunes
- [ ] Usar en código
- [ ] Documentar

### Task 0.2.8: numpy.typing

- **Descripción:** Usar `numpy.typing` para tipado preciso de arrays
- **Entregable:** Código usando numpy.typing
- **Prioridad:** Alta
- **Estimación:** 3 horas

```python
import numpy as np
import numpy.typing as npt

def process_float_array(
    arr: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Procesa array de float64."""
    return arr * 2.0

def process_generic_numeric(
    arr: npt.NDArray[np.number]
) -> npt.NDArray[np.number]:
    """Procesa array numérico genérico."""
    return arr + 1
```

**Checklist:**
- [ ] Reemplazar `np.ndarray` por tipos específicos
- [ ] Usar `npt.NDArray` con dtype
- [ ] Actualizar signatures
- [ ] Actualizar documentación

## Criterios de Aceptación

- [ ] mypy en modo strict sin errores
- [ ] No hay `Any` innecesarios
- [ ] Generics usados correctamente
- [ ] Type guards implementados donde corresponde
- [ ] numpy.typing usado consistentemente
- [ ] Documentación de tipos completa

## Estimación Total

**20 horas** (~2.5 días de desarrollo)

## Referencias

- Python Type Hints: https://docs.python.org/3/library/typing.html
- mypy documentation: https://mypy.readthedocs.io/
- numpy.typing: https://numpy.org/doc/stable/reference/typing.html
