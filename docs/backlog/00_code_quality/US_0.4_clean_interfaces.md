# User Story 0.4: Interfaces Limpias y Result Classes

**Epic**: Code Quality Foundation
**Priority**: CRITICAL
**Estimated Time**: 14 horas
**Status**: 📋 Pending

## User Story

Como desarrollador de la biblioteca,
Quiero que todas las interfaces retornen tipos claros y fuertemente tipados,
Para que los usuarios tengan autocompletado en el IDE y no tengan que adivinar qué contienen los retornos.

## Context

Actualmente tenemos varios puntos donde las interfaces retornan tipos confusos:

### Problemas Identificados (8 casos detectados por validación)

1. **Dict[str, Any] en retornos** (5 casos):
   - `optimization_numérica_avanzada/models/optimization_models.py:36` - `to_dict()`
   - `linalg/models/models.py:55` - `memory_layout_info()`
   - `visualization/models/models.py:49` - `additional_params()`
   - `core/handlers/config_handler.py:80` - `deep_update()`
   - `core/handlers/config_handler.py:101` - `get_section()`

2. **Tuplas largas en retornos** (3 casos):
   - `linalg/services/linalg.py:102` - LU decomposition
   - `linalg/services/linalg.py:120` - SVD decomposition
   - `linalg/services/services.py:158` - LU factorization

3. **get_params() en interfaces** (no detectado automáticamente):
   - `core/interfaces/estimator_interface.py:28`
   - `core/interfaces/optimizer_interface.py:24`

## Acceptance Criteria

### ✅ Criterio 1: Clases de Result para Descomposiciones

- [ ] Crear `ml_lib/linalg/models/results.py` con:
  - [ ] `QRDecompositionResult` con validación
  - [ ] `LUDecompositionResult` con validación
  - [ ] `SVDDecompositionResult` con validación
  - [ ] `EigenDecompositionResult` con validación
  - [ ] `CholeskyDecompositionResult` si existe

- [ ] Actualizar servicios de linalg para usar result classes:
  - [ ] `linalg/services/linalg.py` - métodos de descomposición
  - [ ] `linalg/services/services.py` - funciones auxiliares
  - [ ] Mantener métodos legacy con `@deprecated` si es necesario

- [ ] Tests para result classes:
  - [ ] Test validación de dimensiones en `__post_init__`
  - [ ] Test propiedades matemáticas (Q @ R == A, etc.)

### ✅ Criterio 2: Config Classes para Estimadores/Optimizadores

- [ ] Crear `ml_lib/core/models/configs.py` con:
  - [ ] `EstimatorConfig` (Protocol base)
  - [ ] `OptimizerConfig` (Protocol base)

- [ ] Crear `ml_lib/optimization/models/configs.py` con configs específicos:
  - [ ] `SGDConfig` con validación
  - [ ] `AdamConfig` con validación
  - [ ] `AdamWConfig` con validación
  - [ ] Otros optimizadores según necesidad

- [ ] Actualizar interfaces core:
  - [ ] `EstimatorInterface`: añadir `get_config()` y `set_config()`
  - [ ] `OptimizerInterface`: añadir `get_config()` y `set_config()`
  - [ ] Mantener `get_params()` como wrapper de compatibilidad sklearn

- [ ] Tests para config classes:
  - [ ] Test validación en `__post_init__`
  - [ ] Test conversión config <-> dict para compatibilidad

### ✅ Criterio 3: Clases Específicas para Metadatos

- [ ] Refactorizar `linalg/models/models.py:55`:
  - [ ] Crear `MemoryLayoutInfo` dataclass
  - [ ] Reemplazar `memory_layout_info()` para retornar la clase
  - [ ] Mantener propiedad legacy si es necesario

- [ ] Refactorizar `visualization/models/models.py:49`:
  - [ ] Ya existe `_additional_params` privado - correcto
  - [ ] Documentar que `additional_params` es backward compatibility
  - [ ] Considerar deprecar en futuras versiones

- [ ] Refactorizar `optimization_numérica_avanzada/models/optimization_models.py:36`:
  - [ ] `to_dict()` es legítimo para serialización
  - [ ] Añadir método complementario `from_dict()` si no existe
  - [ ] Documentar que es para serialización, no para uso general

### ✅ Criterio 4: ConfigHandler - Casos Legítimos de Dict

- [ ] Revisar `core/handlers/config_handler.py`:
  - [ ] `deep_update(Dict, Dict) -> Dict` - legítimo (utilidad genérica)
  - [ ] `get_section(str) -> Dict[str, Any]` - considerar retornar clase Config
  - [ ] Documentar claramente cuándo usar ConfigHandler vs Config classes

- [ ] Si get_section debe retornar dict genérico:
  - [ ] Añadir docstring explicando por qué
  - [ ] Proveer método alternativo tipado si es posible

### ✅ Criterio 5: Actualización de Documentación

- [ ] Actualizar `docs/architecture/INTERFACE_IMPROVEMENTS.md`:
  - [ ] Añadir ejemplos de uso de result classes
  - [ ] Añadir ejemplos de uso de config classes

- [ ] Añadir ejemplos en docstrings:
  - [ ] Example para usar `QRDecompositionResult`
  - [ ] Example para usar config classes

- [ ] Migration guide:
  - [ ] Cómo migrar de tuplas a result classes
  - [ ] Cómo migrar de get_params() a get_config()

### ✅ Criterio 6: Validación Automática Pasa

- [ ] `python3 scripts/check_module_structure.py` debe reportar:
  - [ ] 0 warnings en módulos refactorizados
  - [ ] Warnings solo en código legacy explícitamente marcado

## Tasks

### Fase 1: Crear Result Classes (3 horas)

**Task 0.4.1**: Crear result classes para linalg

```bash
# Crear archivo
touch ml_lib/linalg/models/results.py
```

```python
# ml_lib/linalg/models/results.py
"""Result classes para operaciones de álgebra lineal."""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class QRDecompositionResult:
    """Resultado de la descomposición QR.

    Atributos:
        Q: Matriz ortogonal (m x n)
        R: Matriz triangular superior (n x n)

    La descomposición QR factoriza una matriz A en:
        A = Q @ R

    donde Q es ortogonal (Q.T @ Q = I) y R es triangular superior.
    """
    Q: np.ndarray
    R: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones y propiedades."""
        if self.Q.ndim != 2:
            raise ValueError(f"Q debe ser 2D, got {self.Q.ndim}D")
        if self.R.ndim != 2:
            raise ValueError(f"R debe ser 2D, got {self.R.ndim}D")
        if self.Q.shape[1] != self.R.shape[0]:
            raise ValueError(
                f"Dimensiones incompatibles: Q.shape[1]={self.Q.shape[1]} "
                f"!= R.shape[0]={self.R.shape[0]}"
            )

    def reconstruct(self) -> np.ndarray:
        """Reconstruye la matriz original A = Q @ R."""
        return self.Q @ self.R

    def verify_orthogonality(self, tol: float = 1e-10) -> bool:
        """Verifica que Q sea ortogonal (Q.T @ Q ≈ I)."""
        product = self.Q.T @ self.Q
        identity = np.eye(self.Q.shape[1])
        return np.allclose(product, identity, atol=tol)


@dataclass
class LUDecompositionResult:
    """Resultado de la descomposición LU con pivoting.

    Atributos:
        L: Matriz triangular inferior con diagonal de 1s
        U: Matriz triangular superior
        P: Matriz de permutación

    La descomposición LU con pivoting factoriza:
        PA = LU

    donde P es una permutación, L es triangular inferior con diagonal
    unitaria y U es triangular superior.
    """
    L: np.ndarray
    U: np.ndarray
    P: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.L.ndim != 2 or self.U.ndim != 2 or self.P.ndim != 2:
            raise ValueError("L, U, P deben ser matrices 2D")

        n = self.L.shape[0]
        if not (self.L.shape == (n, n) and self.U.shape == (n, n) and self.P.shape == (n, n)):
            raise ValueError(
                f"L, U, P deben ser cuadradas del mismo tamaño. "
                f"Got L:{self.L.shape}, U:{self.U.shape}, P:{self.P.shape}"
            )

    def reconstruct(self) -> np.ndarray:
        """Reconstruye PA = LU."""
        return self.L @ self.U

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Resuelve Ax = b usando la descomposición LU.

        Args:
            b: Vector del lado derecho

        Returns:
            Solución x del sistema Ax = b
        """
        # PA = LU, entonces Ax = b => PAx = Pb => LUx = Pb
        Pb = self.P @ b
        # Forward substitution: Ly = Pb
        y = self._forward_substitution(self.L, Pb)
        # Backward substitution: Ux = y
        x = self._backward_substitution(self.U, y)
        return x

    @staticmethod
    def _forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resolución forward para Ly = b."""
        n = L.shape[0]
        y = np.zeros_like(b)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        return y

    @staticmethod
    def _backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resolución backward para Ux = b."""
        n = U.shape[0]
        x = np.zeros_like(b)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        return x


@dataclass
class SVDDecompositionResult:
    """Resultado de la descomposición en valores singulares (SVD).

    Atributos:
        U: Matriz de vectores singulares izquierdos (m x m)
        s: Vector de valores singulares (min(m,n),) en orden descendente
        Vt: Matriz de vectores singulares derechos transpuesta (n x n)

    La descomposición SVD factoriza:
        A = U @ S @ Vt

    donde U y Vt son ortogonales y S es diagonal con valores singulares.
    """
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.U.ndim != 2 or self.Vt.ndim != 2:
            raise ValueError("U y Vt deben ser matrices 2D")
        if self.s.ndim != 1:
            raise ValueError(f"s debe ser vector 1D, got {self.s.ndim}D")

    def reconstruct(self, full_matrices: bool = True) -> np.ndarray:
        """Reconstruye la matriz original A = U @ S @ Vt.

        Args:
            full_matrices: Si True, usa U y Vt completas.
                          Si False, usa solo los primeros k vectores singulares.

        Returns:
            Matriz reconstruida
        """
        k = len(self.s)
        S = np.zeros((self.U.shape[0], self.Vt.shape[0]))
        S[:k, :k] = np.diag(self.s)
        return self.U @ S @ self.Vt

    def low_rank_approximation(self, rank: int) -> np.ndarray:
        """Calcula aproximación de bajo rango de A.

        Args:
            rank: Número de valores singulares a mantener

        Returns:
            Aproximación de rango 'rank' de A
        """
        if rank > len(self.s):
            rank = len(self.s)

        U_k = self.U[:, :rank]
        s_k = self.s[:rank]
        Vt_k = self.Vt[:rank, :]

        S_k = np.diag(s_k)
        return U_k @ S_k @ Vt_k

    def condition_number(self) -> float:
        """Calcula el número de condición de la matriz."""
        return self.s[0] / self.s[-1] if self.s[-1] != 0 else np.inf


@dataclass
class EigenDecompositionResult:
    """Resultado de la descomposición en valores propios.

    Atributos:
        eigenvalues: Vector de valores propios
        eigenvectors: Matriz de vectores propios como columnas

    Para una matriz A:
        A @ v = λ @ v

    donde λ es un valor propio y v es su vector propio correspondiente.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    def __post_init__(self):
        """Validación de dimensiones."""
        if self.eigenvalues.ndim != 1:
            raise ValueError(f"eigenvalues debe ser 1D, got {self.eigenvalues.ndim}D")
        if self.eigenvectors.ndim != 2:
            raise ValueError(f"eigenvectors debe ser 2D, got {self.eigenvectors.ndim}D")

        if len(self.eigenvalues) != self.eigenvectors.shape[1]:
            raise ValueError(
                f"Número de eigenvalues ({len(self.eigenvalues)}) debe coincidir "
                f"con número de eigenvectors ({self.eigenvectors.shape[1]})"
            )

    def reconstruct(self) -> np.ndarray:
        """Reconstruye la matriz si es diagonalizable: A = V @ D @ V^(-1)."""
        V = self.eigenvectors
        D = np.diag(self.eigenvalues)
        return V @ D @ np.linalg.inv(V)

    def dominant_eigenvalue(self) -> float:
        """Retorna el valor propio dominante (mayor valor absoluto)."""
        return self.eigenvalues[np.argmax(np.abs(self.eigenvalues))]
```

Estimación: 3 horas (incluyendo tests básicos)

### Fase 2: Actualizar Servicios de Linalg (2 horas)

**Task 0.4.2**: Actualizar métodos de descomposición

- Actualizar `linalg/services/linalg.py` para usar result classes
- Actualizar `linalg/services/services.py`
- Añadir wrappers legacy si es necesario con `@deprecated`

Estimación: 2 horas

### Fase 3: Crear Config Classes (3 horas)

**Task 0.4.3**: Crear system de configuración tipado

- Crear protocolos base en `core/models/configs.py`
- Crear configs específicos en `optimization/models/configs.py`
- Implementar métodos de conversión dict <-> config

Estimación: 3 horas

### Fase 4: Actualizar Interfaces Core (2 horas)

**Task 0.4.4**: Añadir get_config/set_config a interfaces

- Modificar `EstimatorInterface` y `OptimizerInterface`
- Actualizar docstrings con ejemplos
- Mantener backward compatibility con get_params()

Estimación: 2 horas

### Fase 5: Refactorizar Metadatos (2 horas)

**Task 0.4.5**: Limpiar Dict[str, Any] en modelos

- Crear `MemoryLayoutInfo` en linalg
- Documentar casos legítimos de Dict (como `to_dict()` para serialización)
- Actualizar ConfigHandler si es necesario

Estimación: 2 horas

### Fase 6: Tests y Validación (2 horas)

**Task 0.4.6**: Tests completos para nuevas clases

- Tests de validación en `__post_init__`
- Tests de propiedades matemáticas (reconstrucción, ortogonalidad, etc.)
- Tests de conversión config <-> dict
- Ejecutar validación automática

Estimación: 2 horas

## Definition of Done

- [ ] ✅ Todas las descomposiciones de linalg retornan result classes
- [ ] ✅ EstimatorInterface y OptimizerInterface tienen get_config()
- [ ] ✅ Al menos 2 configs específicos implementados (SGD, Adam)
- [ ] ✅ Dict[str, Any] solo en casos legítimos y documentados
- [ ] ✅ Tests de validación pasan
- [ ] ✅ Tests de propiedades matemáticas pasan
- [ ] ✅ Script de validación reporta 0 warnings en código refactorizado
- [ ] ✅ Documentación actualizada con ejemplos
- [ ] ✅ Migration guide creado
- [ ] ✅ mypy sin errores
- [ ] ✅ Commit y push exitoso

## Dependencies

- Depende de: US 0.1 (Refactorización a clases) - parcialmente completado
- Blocked by: Ninguno
- Blocks: US 0.5 (si existe) o desarrollo de nuevos módulos

## Notes

### Decisiones de Diseño

1. **Usar dataclasses para result classes**: Más ligeras que clases regulares, validación en `__post_init__`

2. **Mantener backward compatibility**: get_params() sigue existiendo como wrapper

3. **No fallar validación por Dict legítimos**: ConfigHandler y to_dict() son casos válidos

### Casos Especiales

- `ConfigHandler.get_section()`: Considerar si debe retornar Config class o Dict genérico
- `to_dict()` methods: Legítimos para serialización, no para uso general de API

### Referencias

- `docs/architecture/INTERFACE_IMPROVEMENTS.md` - Diseño completo
- `docs/CODE_QUALITY_GUIDELINES.md` - Principios generales
- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
