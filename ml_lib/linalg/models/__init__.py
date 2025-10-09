"""
__init__.py para el submódulo models de linalg
"""

from .models import (
    Matrix,
    SparseMatrix,
    LinearSystemSolution,
    DecompositionResult,
    EigenDecomposition,
)
from .enums import (
    MatrixPrecision,
    MemoryLayout,
    DecompositionMethod,
    SolverMethod,
    SparseFormat,
    NormType,
    EigenSolver,
    MatrixProperty,
    PreconditionerType,
)
from .results import (
    QRDecompositionResult,
    LUDecompositionResult,
    SVDDecompositionResult,
    EigenDecompositionResult,
    CholeskyDecompositionResult,
)

__all__ = [
    # Models
    "Matrix",
    "SparseMatrix",
    "LinearSystemSolution",
    "DecompositionResult",
    "EigenDecomposition",
    # Enums
    "MatrixPrecision",
    "MemoryLayout",
    "DecompositionMethod",
    "SolverMethod",
    "SparseFormat",
    "NormType",
    "EigenSolver",
    "MatrixProperty",
    "PreconditionerType",
    # Result Classes
    "QRDecompositionResult",
    "LUDecompositionResult",
    "SVDDecompositionResult",
    "EigenDecompositionResult",
    "CholeskyDecompositionResult",
]
