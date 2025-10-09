"""
__init__.py para el módulo interfaces de linalg
"""
from .matrix_operation_interface import (
    MatrixOperationInterface,
    DecompositionInterface,
    SolverInterface
)


__all__ = [
    'MatrixOperationInterface',
    'DecompositionInterface',
    'SolverInterface'
]