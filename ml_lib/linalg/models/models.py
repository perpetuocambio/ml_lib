"""
Modelos para álgebra lineal en ml_lib
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class Matrix:
    """Representación de una matriz con metadatos."""

    data: np.ndarray
    shape: Tuple[int, ...] = field(init=False)
    dtype: np.dtype = field(init=False)
    is_sparse: bool = False
    layout: str = "C"  # C-contiguous o F-contiguous
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None

    def __post_init__(self):
        """Inicialización posterior a la creación."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data)

        self.shape = self.data.shape
        self.dtype = self.data.dtype

        # Determinar layout
        if self.data.flags.c_contiguous:
            self.layout = "C"
        elif self.data.flags.f_contiguous:
            self.layout = "F"
        else:
            self.layout = "unknown"

    def validate_matrix(self) -> None:
        """Valida la integridad de la matriz."""
        if np.any(np.isnan(self.data)):
            raise ValueError("Matriz contiene valores NaN")
        if np.any(np.isinf(self.data)):
            raise ValueError("Matriz contiene valores infinitos")

    def is_square(self) -> bool:
        """Verifica si la matriz es cuadrada."""
        return len(self.shape) == 2 and self.shape[0] == self.shape[1]

    def is_symmetric(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Verifica si la matriz es simétrica."""
        if not self.is_square():
            return False
        return np.allclose(self.data, self.data.T, rtol=rtol, atol=atol)

    def memory_layout_info(self) -> Dict[str, Any]:
        """Información sobre el layout de memoria."""
        return {
            "c_contiguous": self.data.flags.c_contiguous,
            "f_contiguous": self.data.flags.f_contiguous,
            "layout": self.layout,
            "nbytes": self.data.nbytes,
            "itemsize": self.data.itemsize,
        }


@dataclass
class SparseMatrix:
    """Representación de una matriz dispersa."""

    data: np.ndarray
    row_indices: np.ndarray
    col_indices: np.ndarray
    shape: Tuple[int, int]
    format: str = "COO"  # CSR, CSC, COO
    nnz: int = field(init=False)  # Número de elementos no nulos
    density: float = field(init=False)  # Densidad de la matriz

    def __post_init__(self):
        """Inicialización posterior a la creación."""
        self.nnz = len(self.data)
        self.density = self.nnz / (self.shape[0] * self.shape[1])

        if len(self.row_indices) != self.nnz or len(self.col_indices) != self.nnz:
            raise ValueError("Data y arrays de índices deben tener la misma longitud")

        if self.format not in ["CSR", "CSC", "COO"]:
            raise ValueError(f"Formato disperso no soportado: {self.format}")


@dataclass
class DecompositionResult:
    """Resultado de una descomposición matricial."""

    components: Tuple[np.ndarray, ...]
    algorithm: str
    computation_time: float
    memory_used: float
    error: Optional[float] = None
    converged: bool = True
    iterations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Inicialización posterior a la creación."""
        self.metadata["created_at"] = self._get_timestamp()

    def _get_timestamp(self) -> str:
        """Obtiene marca de tiempo."""
        from datetime import datetime

        return datetime.now().isoformat()


@dataclass
class LinearSystemSolution:
    """Solución de un sistema lineal."""

    solution: np.ndarray
    residual: np.ndarray
    method: str
    condition_number: float
    iterations: Optional[int] = None
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Inicialización posterior a la creación."""
        self.metadata["created_at"] = self._get_timestamp()

    def _get_timestamp(self) -> str:
        """Obtiene marca de tiempo."""
        from datetime import datetime

        return datetime.now().isoformat()


@dataclass
class EigenDecomposition:
    """Descomposición de valores propios."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    method: str
    computation_time: float
    memory_used: float
    converged: bool = True
    iterations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Inicialización posterior a la creación."""
        self.metadata["created_at"] = self._get_timestamp()

    def _get_timestamp(self) -> str:
        """Obtiene marca de tiempo."""
        from datetime import datetime

        return datetime.now().isoformat()


@dataclass
class MatrixOperationConfig:
    """Configuración para operaciones matriciales."""

    precision: str = "double"  # single, double, extended
    threading: bool = True
    num_threads: int = 4
    cache_blocking: bool = True
    block_size: int = 64
    memory_alignment: int = 32
    optimization_level: str = "O2"  # O0, O1, O2, O3
    use_simd: bool = True
    simd_instruction_set: str = "AVX2"  # SSE, AVX, AVX2, AVX512
    custom_params: Dict[str, Any] = field(default_factory=dict)
