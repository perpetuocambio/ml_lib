"""
Interfaces para operaciones de álgebra lineal en ml_lib
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..models import (
        QRDecompositionResult,
        LUDecompositionResult,
        SVDDecompositionResult,
        CholeskyDecompositionResult,
    )

T = TypeVar("T", bound=np.ndarray)
MatrixType = TypeVar("MatrixType", bound="Matrix")


class MatrixOperationInterface(ABC, Generic[T]):
    """Interface base para operaciones matriciales."""

    @abstractmethod
    def matmul(self, A: T, B: T) -> T:
        """Producto matricial A @ B."""
        pass

    @abstractmethod
    def solve(self, A: T, b: T) -> T:
        """Resuelve el sistema lineal Ax = b."""
        pass

    @abstractmethod
    def inv(self, A: T) -> T:
        """Calcula la inversa de la matriz A."""
        pass

    @abstractmethod
    def det(self, A: T) -> float:
        """Calcula el determinante de la matriz A."""
        pass


class DecompositionInterface(ABC, Generic[T]):
    """Interface base para descomposiciones matriciales."""

    @abstractmethod
    def decompose(self, A: T) -> "SVDDecompositionResult":
        """Realiza la descomposición SVD de la matriz A."""
        pass

    @abstractmethod
    def reconstruct(self, result: "SVDDecompositionResult") -> T:
        """Reconstruye la matriz desde sus componentes SVD."""
        pass


class SolverInterface(ABC, Generic[T]):
    """Interface base para solvers lineales."""

    @abstractmethod
    def solve_linear_system(self, A: T, b: T) -> T:
        """Resuelve el sistema lineal Ax = b."""
        pass

    @abstractmethod
    def solve_least_squares(self, A: T, b: T) -> T:
        """Resuelve el problema de mínimos cuadrados."""
        pass


class BLASInterface(ABC, Generic[T]):
    """Interface para operaciones BLAS optimizadas."""

    @abstractmethod
    def gemm(self, alpha: float, A: T, B: T, beta: float = 0.0, C: T = None) -> T:
        """Producto matricial general (GEMM)."""
        pass

    @abstractmethod
    def gemv(self, alpha: float, A: T, x: T, beta: float = 0.0, y: T = None) -> T:
        """Producto matriz-vector (GEMV)."""
        pass


class LAPACKInterface(ABC, Generic[T]):
    """Interface para operaciones LAPACK."""

    @abstractmethod
    def qr_factorize(self, A: T) -> "QRDecompositionResult":
        """Factorización QR de la matriz A."""
        pass

    @abstractmethod
    def lu_factorize(self, A: T) -> "LUDecompositionResult":
        """Factorización LU de la matriz A."""
        pass

    @abstractmethod
    def cholesky_factorize(self, A: T) -> "CholeskyDecompositionResult":
        """Factorización de Cholesky de la matriz A."""
        pass
