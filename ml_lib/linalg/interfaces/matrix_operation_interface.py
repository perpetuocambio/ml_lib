"""
Interfaz base para operaciones de álgebra lineal con tipado estricto.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union
import numpy as np


T = TypeVar('T', bound=np.ndarray)


class MatrixOperationInterface(ABC, Generic[T]):
    """Interface base para operaciones de matriz."""
    
    @abstractmethod
    def matmul(self, A: T, B: T) -> T:
        """Producto matricial."""
        pass
    
    @abstractmethod
    def solve(self, A: T, b: T) -> T:
        """Resuelve el sistema lineal Ax = b."""
        pass
    
    @abstractmethod
    def inv(self, A: T) -> T:
        """Calcula la inversa de la matriz."""
        pass
    
    @abstractmethod
    def det(self, A: T) -> float:
        """Calcula el determinante de la matriz."""
        pass


class DecompositionInterface(ABC, Generic[T]):
    """Interface base para descomposiciones matriciales."""
    
    @abstractmethod
    def decompose(self, A: T) -> tuple:
        """Realiza la descomposición de la matriz."""
        pass
    
    @abstractmethod
    def reconstruct(self, *components) -> T:
        """Reconstruye la matriz desde sus componentes."""
        pass


class SolverInterface(ABC, Generic[T]):
    """Interface base para solvers lineales."""
    
    @abstractmethod
    def solve(self, A: T, b: T) -> T:
        """Resuelve el sistema lineal Ax = b."""
        pass
    
    @abstractmethod
    def solve_least_squares(self, A: T, b: T) -> T:
        """Resuelve el problema de mínimos cuadrados."""
        pass