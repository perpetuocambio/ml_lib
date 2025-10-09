"""
Servicios para operaciones de álgebra lineal en ml_lib
"""

import numpy as np
from typing import Optional, Tuple
import logging
from scipy.linalg import blas
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

# Importar modelos e interfaces
from .models import (
    SparseMatrix,
)
from .interfaces import (
    MatrixOperationInterface,
    DecompositionInterface,
    SolverInterface,
    BLASInterface,
    LAPACKInterface,
)


class BLASService(BLASInterface[np.ndarray]):
    """Servicio para operaciones BLAS optimizadas."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def gemm(
        alpha: float,
        A: np.ndarray,
        B: np.ndarray,
        beta: float = 0.0,
        C: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Producto matricial general (GEMM): C = alpha * A * B + beta * C.

        Args:
            alpha: Factor escalar para A * B
            A: Matriz izquierda (m x k)
            B: Matriz derecha (k x n)
            beta: Factor escalar para C
            C: Matriz de salida opcional (m x n)

        Returns:
            Matriz resultante
        """
        # Validar formas de matrices
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A y B deben ser matrices 2D")

        m, k_A = A.shape
        k_B, n = B.shape

        if k_A != k_B:
            raise ValueError(
                f"Dimensiones incompatibles: A es {A.shape}, B es {B.shape}"
            )

        # Crear matriz C si no se proporciona
        if C is None:
            C = np.zeros((m, n), dtype=np.result_type(A.dtype, B.dtype))
        elif C.shape != (m, n):
            raise ValueError(
                f"Forma de C incorrecta: esperada {(m, n)}, obtenida {C.shape}"
            )

        # Determinar tipo de precisión
        if A.dtype == np.float64 and B.dtype == np.float64:
            # Usar BLAS de doble precisión
            return blas.dgemm(alpha, A, B, beta, C)
        elif A.dtype == np.float32 and B.dtype == np.float32:
            # Usar BLAS de simple precisión
            return blas.sgemm(alpha, A, B, beta, C)
        else:
            # Fallback a numpy para otros tipos
            return alpha * (A @ B) + beta * C

    @staticmethod
    def gemv(
        alpha: float,
        A: np.ndarray,
        x: np.ndarray,
        beta: float = 0.0,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Producto matriz-vector (GEMV): y = alpha * A * x + beta * y.

        Args:
            alpha: Factor escalar para A * x
            A: Matriz (m x n)
            x: Vector (n,)
            beta: Factor escalar para y
            y: Vector de salida opcional (m,)

        Returns:
            Vector resultante
        """
        # Validar formas
        if A.ndim != 2:
            raise ValueError("A debe ser una matriz 2D")
        if x.ndim != 1:
            raise ValueError("x debe ser un vector 1D")

        m, n = A.shape
        if len(x) != n:
            raise ValueError(
                f"Dimensiones incompatibles: A es {A.shape}, x es {x.shape}"
            )

        # Crear vector y si no se proporciona
        if y is None:
            y = np.zeros(m, dtype=np.result_type(A.dtype, x.dtype))
        elif len(y) != m:
            raise ValueError(
                f"Forma de y incorrecta: esperada {(m,)}, obtenida {y.shape}"
            )

        # Determinar tipo de precisión
        if A.dtype == np.float64 and x.dtype == np.float64:
            return blas.dgemv(alpha, A, x, beta, y)
        elif A.dtype == np.float32 and x.dtype == np.float32:
            return blas.sgemv(alpha, A, x, beta, y)
        else:
            # Fallback a numpy
            return alpha * (A @ x) + beta * y


class LAPACKService(LAPACKInterface[np.ndarray]):
    """Servicio para operaciones LAPACK."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def qr_factorize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorización QR de la matriz A.

        Args:
            A: Matriz de entrada (m x n)

        Returns:
            Tupla (Q, R) donde A = Q @ R
        """
        if A.ndim != 2:
            raise ValueError("A debe ser una matriz 2D")

        # Usar scipy.linalg.qr
        Q, R = np.linalg.qr(A)
        return Q, R

    @staticmethod
    def lu_factorize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Factorización LU de la matriz A.

        Args:
            A: Matriz de entrada (n x n)

        Returns:
            Tupla (P, L, U) donde P @ A = L @ U
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A debe ser una matriz cuadrada 2D")

        # Usar scipy.linalg.lu
        P, L, U = np.linalg.lu(A)
        return P, L, U

    @staticmethod
    def cholesky_factorize(A: np.ndarray) -> np.ndarray:
        """
        Factorización de Cholesky de la matriz A.

        Args:
            A: Matriz simétrica definida positiva (n x n)

        Returns:
            Matriz triangular inferior L tal que A = L @ L.T
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A debe ser una matriz cuadrada 2D")

        # Verificar que sea simétrica
        if not np.allclose(A, A.T):
            raise ValueError("A debe ser simétrica")

        # Usar scipy.linalg.cholesky
        L = np.linalg.cholesky(A)
        return L


class MatrixOperationService(MatrixOperationInterface[np.ndarray]):
    """Servicio para operaciones matriciales generales."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Producto matricial A @ B."""
        return A @ B

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve el sistema lineal Ax = b."""
        return np.linalg.solve(A, b)

    def inv(self, A: np.ndarray) -> np.ndarray:
        """Calcula la inversa de la matriz A."""
        return np.linalg.inv(A)

    def det(self, A: np.ndarray) -> float:
        """Calcula el determinante de la matriz A."""
        return np.linalg.det(A)


class DecompositionService(DecompositionInterface[np.ndarray]):
    """Servicio para descomposiciones matriciales."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def decompose(self, A: np.ndarray) -> tuple:
        """Realiza descomposición SVD de la matriz A."""
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        return (U, s, Vt)

    def reconstruct(self, U: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> np.ndarray:
        """Reconstruye la matriz desde sus componentes SVD."""
        # Reconstruir matriz usando SVD
        S = np.zeros((U.shape[1], Vt.shape[0]))
        np.fill_diagonal(S, s)
        return U @ S @ Vt


class SolverService(SolverInterface[np.ndarray]):
    """Servicio para solvers lineales."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve el sistema lineal Ax = b."""
        return np.linalg.solve(A, b)

    def solve_least_squares(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Resuelve el problema de mínimos cuadrados."""
        return np.linalg.lstsq(A, b, rcond=None)[0]


class SparseMatrixService:
    """Servicio para operaciones con matrices dispersas."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def create_sparse_matrix(
        data: np.ndarray,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        shape: Tuple[int, int],
        format: str = "COO",
    ) -> SparseMatrix:
        """
        Crea una matriz dispersa.

        Args:
            data: Valores no nulos
            row_indices: Índices de filas
            col_indices: Índices de columnas
            shape: Forma de la matriz (filas, columnas)
            format: Formato disperso (COO, CSR, CSC)

        Returns:
            Objeto SparseMatrix
        """
        return SparseMatrix(
            data=data,
            row_indices=row_indices,
            col_indices=col_indices,
            shape=shape,
            format=format,
        )

    @staticmethod
    def sparse_matmul(A: SparseMatrix, B: np.ndarray) -> np.ndarray:
        """
        Producto matricial entre matriz dispersa y densa.

        Args:
            A: Matriz dispersa
            B: Matriz densa

        Returns:
            Resultado del producto matricial
        """
        # Convertir a formato scipy.sparse según el formato
        if A.format == "COO":
            sparse_A = coo_matrix(
                (A.data, (A.row_indices, A.col_indices)), shape=A.shape
            )
        elif A.format == "CSR":
            sparse_A = csr_matrix((A.data, A.row_indices, A.col_indices), shape=A.shape)
        elif A.format == "CSC":
            sparse_A = csc_matrix((A.data, A.row_indices, A.col_indices), shape=A.shape)
        else:
            raise ValueError(f"Formato no soportado: {A.format}")

        # Realizar multiplicación
        return sparse_A @ B
