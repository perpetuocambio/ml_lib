"""
__init__.py para el módulo de álgebra lineal de ml_lib
"""

from .interfaces import (
    MatrixOperationInterface,
    DecompositionInterface,
    SolverInterface,
    BLASInterface,
    LAPACKInterface,
)

from .models import (
    Matrix,
    SparseMatrix,
    DecompositionResult,
    LinearSystemSolution,
    EigenDecomposition,
    MatrixOperationConfig,
)

from .services import (
    BLASService,
    LAPACKService,
    MatrixOperationService,
    DecompositionService,
    SolverService,
    SparseMatrixService,
)

from .handlers import (
    LinearAlgebraErrorHandler,
    MatrixConfigHandler,
    MemoryLayoutHandler,
    PrecisionHandler,
    MatrixValidationHandler,
)

from .linalg import LinearAlgebraEngine, LinearAlgebraFactory


__all__ = [
    # Interfaces
    "MatrixOperationInterface",
    "DecompositionInterface",
    "SolverInterface",
    "BLASInterface",
    "LAPACKInterface",
    # Modelos
    "Matrix",
    "SparseMatrix",
    "DecompositionResult",
    "LinearSystemSolution",
    "EigenDecomposition",
    "MatrixOperationConfig",
    # Servicios
    "BLASService",
    "LAPACKService",
    "MatrixOperationService",
    "DecompositionService",
    "SolverService",
    "SparseMatrixService",
    # Handlers
    "LinearAlgebraErrorHandler",
    "MatrixConfigHandler",
    "MemoryLayoutHandler",
    "PrecisionHandler",
    "MatrixValidationHandler",
    # Implementaciones
    "LinearAlgebraEngine",
    "LinearAlgebraFactory",
]
