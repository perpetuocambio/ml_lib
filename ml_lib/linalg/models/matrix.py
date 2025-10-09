"""
Modelos para operaciones de álgebra lineal con tipado estricto.
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Inicialización post-creación."""
        if not isinstance(self.data, np.ndarray):
            self.data = np.asarray(self.data)
        
        self.shape = self.data.shape
        self.dtype = self.data.dtype
    
    def validate_matrix(self) -> None:
        """Valida la integridad de la matriz."""
        if np.any(np.isnan(self.data)):
            raise ValueError("Matrix contains NaN values")
        if np.any(np.isinf(self.data)):
            raise ValueError("Matrix contains infinite values")


@dataclass
class SparseMatrix:
    """Representación de una matriz dispersa."""
    
    data: np.ndarray
    row_indices: np.ndarray
    col_indices: np.ndarray
    shape: Tuple[int, int]
    format: str = "COO"  # CSR, CSC, COO
    nnz: int = field(init=False)  # Número de elementos no nulos
    
    def __post_init__(self) -> None:
        """Inicialización post-creación."""
        self.nnz = len(self.data)
        
        if len(self.row_indices) != self.nnz or len(self.col_indices) != self.nnz:
            raise ValueError("Data and index arrays must have the same length")
        
        if self.format not in ["CSR", "CSC", "COO"]:
            raise ValueError(f"Unsupported sparse format: {self.format}")


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