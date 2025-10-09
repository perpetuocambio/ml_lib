"""
__init__.py para el módulo linalg
"""
from .interfaces import *
from .models import *
from .services import *


__all__ = [
    # Interfaces
    'MatrixOperationInterface',
    'DecompositionInterface',
    'SolverInterface',
    
    # Models
    'Matrix',
    'SparseMatrix',
    'DecompositionResult',
    
    # Services
    'BLASService',
    'DecompositionService',
    'SparseService'
]