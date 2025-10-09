"""
Implementación principal del módulo de álgebra lineal para ml_lib
"""

import numpy as np
from typing import Optional, Tuple

# Importar componentes
from .models import (
    Matrix,
    SparseMatrix,
    MatrixOperationConfig,
    QRDecompositionResult,
    LUDecompositionResult,
    SVDDecompositionResult,
    EigenDecompositionResult,
    CholeskyDecompositionResult,
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
        format: str = "COO",
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
        C: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Producto matricial general optimizado."""
        return self.blas_service.gemm(alpha, A, B, beta, C)

    def gemv(
        self,
        alpha: float,
        A: np.ndarray,
        x: np.ndarray,
        beta: float = 0.0,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Producto matriz-vector optimizado."""
        return self.blas_service.gemv(alpha, A, x, beta, y)

    def qr_decomposition(self, A: np.ndarray) -> QRDecompositionResult:
        """Factorización QR."""
        Q, R = self.lapack_service.qr_factorize(A)
        return QRDecompositionResult(Q=Q, R=R)

    def lu_decomposition(self, A: np.ndarray) -> LUDecompositionResult:
        """Factorización LU."""
        P, L, U = self.lapack_service.lu_factorize(A)
        return LUDecompositionResult(L=L, U=U, P=P)

    def cholesky_decomposition(self, A: np.ndarray) -> CholeskyDecompositionResult:
        """Factorización de Cholesky."""
        L = self.lapack_service.cholesky_factorize(A)
        return CholeskyDecompositionResult(L=L)

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve sistema lineal Ax = b."""
        return self.solver_service.solve_linear_system(A, b)

    def solve_least_squares(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve problema de mínimos cuadrados."""
        return self.solver_service.solve_least_squares(A, b)

    def svd_decomposition(
        self, A: np.ndarray, full_matrices: bool = True
    ) -> SVDDecompositionResult:
        """Descomposición SVD."""
        U, s, Vt = np.linalg.svd(A, full_matrices=full_matrices)
        return SVDDecompositionResult(U=U, s=s, Vt=Vt)

    def eigen_decomposition(self, A: np.ndarray) -> EigenDecompositionResult:
        """Descomposición de valores propios."""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        return EigenDecompositionResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    def sparse_matmul(self, A: SparseMatrix, B: np.ndarray) -> np.ndarray:
        """Producto matricial disperso-denso."""
        return self.sparse_service.sparse_matmul(A, B)


class LinearAlgebraFactory:
    """Fábrica para crear componentes de álgebra lineal."""

    @staticmethod
    def create_engine(
        config: Optional[MatrixOperationConfig] = None,
    ) -> LinearAlgebraEngine:
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
        format: str = "COO",
    ) -> SparseMatrix:
        """Crea una matriz dispersa."""
        engine = LinearAlgebraFactory.create_engine()
        return engine.create_sparse_matrix(
            data, row_indices, col_indices, shape, format
        )
