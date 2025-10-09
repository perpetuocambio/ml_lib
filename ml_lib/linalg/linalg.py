"""
Implementación principal del módulo de álgebra lineal para ml_lib
"""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Importar componentes
from .interfaces import (
    MatrixOperationInterface, DecompositionInterface, 
    SolverInterface, BLASInterface, LAPACKInterface
)
from .models import (
    Matrix, SparseMatrix, DecompositionResult, 
    LinearSystemSolution, EigenDecomposition, MatrixOperationConfig
)
from .services import (
    BLASService, LAPACKService, MatrixOperationService,
    DecompositionService, SolverService, SparseMatrixService
)
from .handlers import (
    LinearAlgebraErrorHandler, MatrixConfigHandler, 
    MemoryLayoutHandler, PrecisionHandler, MatrixValidationHandler
)
from ml_lib.core import LoggingService


class LinearAlgebraEngine:
    """Motor principal del módulo de álgebra lineal."""
    
    def __init__(self, config: Optional[MatrixOperationConfig] = None):
        self.config = config or MatrixOperationConfig()
        self.logger_service = LoggingService("LinearAlgebraEngine")
        self.logger = self.logger_service.get_logger()
        
        # Inicializar servicios
        self.error_handler = LinearAlgebraErrorHandler(self.logger)
        self.config_handler = MatrixConfigHandler()
        self.memory_handler = MemoryLayoutHandler()
        self.precision_handler = PrecisionHandler()
        self.validation_handler = MatrixValidationHandler()
        
        # Servicios especializados
        self.blas_service = BLASService(self.logger)
        self.lapack_service = LAPACKService(self.logger)
        self.matrix_op_service = MatrixOperationService(self.logger)
        self.decomp_service = DecompositionService(self.logger)
        self.solver_service = SolverService(self.logger)
        self.sparse_service = SparseMatrixService(self.logger)
        
        self.logger.info("Motor de álgebra lineal inicializado")
    
    def create_matrix(self, data: np.ndarray, **kwargs) -> Matrix:
        """Crea una matriz con metadatos."""
        return Matrix(data=data, **kwargs)
    
    def create_sparse_matrix(
        self,
        data: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        shape: Tuple[int, int],
        format: str = "COO"
    ) -> SparseMatrix:
        """Crea una matriz dispersa."""
        return self.sparse_service.create_sparse_matrix(
            data, row_indices, col_indices, shape, format
        )
    
    def gemm(
        self,
        alpha: float,
        A: np.ndarray,
        B: np.ndarray,
        beta: float = 0.0,
        C: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Producto matricial general optimizado."""
        return self.blas_service.gemm(alpha, A, B, beta, C)
    
    def gemv(
        self,
        alpha: float,
        A: np.ndarray,
        x: np.ndarray,
        beta: float = 0.0,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Producto matriz-vector optimizado."""
        return self.blas_service.gemv(alpha, A, x, beta, y)
    
    def qr_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Factorización QR."""
        return self.lapack_service.qr_factorize(A)
    
    def lu_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Factorización LU."""
        return self.lapack_service.lu_factorize(A)
    
    def cholesky_decomposition(self, A: np.ndarray) -> np.ndarray:
        """Factorización de Cholesky."""
        return self.lapack_service.cholesky_factorize(A)
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve sistema lineal Ax = b."""
        return self.solver_service.solve_linear_system(A, b)
    
    def solve_least_squares(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve problema de mínimos cuadrados."""
        return self.solver_service.solve_least_squares(A, b)
    
    def svd_decomposition(self, A: np.ndarray, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Descomposición SVD."""
        U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
        return U, s, Vt
    
    def eigen_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Descomposición de valores propios."""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return eigenvalues, eigenvectors
    
    def sparse_matmul(self, A: SparseMatrix, B: np.ndarray) -> np.ndarray:
        """Producto matricial disperso-denso."""
        return self.sparse_service.sparse_matmul(A, B)


class LinearAlgebraFactory:
    """Fábrica para crear componentes de álgebra lineal."""
    
    @staticmethod
    def create_engine(config: Optional[MatrixOperationConfig] = None) -> LinearAlgebraEngine:
        """Crea una instancia del motor de álgebra lineal."""
        return LinearAlgebraEngine(config)
    
    @staticmethod
    def create_matrix(data: np.ndarray, **kwargs) -> Matrix:
        """Crea una matriz con metadatos."""
        engine = LinearAlgebraFactory.create_engine()
        return engine.create_matrix(data, **kwargs)
    
    @staticmethod
    def create_sparse_matrix(
        data: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        shape: Tuple[int, int],
        format: str = "COO"
    ) -> SparseMatrix:
        """Crea una matriz dispersa."""
        engine = LinearAlgebraFactory.create_engine()
        return engine.create_sparse_matrix(data, row_indices, col_indices, shape, format)