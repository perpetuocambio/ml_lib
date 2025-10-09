"""
__init__.py para el módulo models de linalg
"""
from .matrix import Matrix, SparseMatrix, DecompositionResult


__all__ = [
    'Matrix',
    'SparseMatrix',
    'DecompositionResult'
]