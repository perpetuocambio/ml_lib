"""
Interfaz para optimizadores con tipado estricto.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar
import numpy as np


T = TypeVar('T', bound=np.ndarray)


class OptimizerInterface(ABC, Generic[T]):
    """Interface base para optimizadores."""
    
    @abstractmethod
    def minimize(
        self,
        func: Callable[[T], float],
        grad: Callable[[T], T],
        initial_params: T
    ) -> T:
        """Minimiza la función objetivo usando gradientes."""
        pass
    
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Obtiene los parámetros del optimizador."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'OptimizerInterface[T]':
        """Establece los parámetros del optimizador."""
        pass


class FirstOrderOptimizerInterface(OptimizerInterface[T], ABC):
    """Interface para optimizadores de primer orden (usando gradientes)."""
    pass


class SecondOrderOptimizerInterface(OptimizerInterface[T], ABC):
    """Interface para optimizadores de segundo orden (usando Hessiana)."""
    
    @abstractmethod
    def minimize_with_hessian(
        self,
        func: Callable[[T], float],
        grad: Callable[[T], T],
        hess: Callable[[T], np.ndarray],
        initial_params: T
    ) -> T:
        """Minimiza usando información de segundo orden (Hessiana)."""
        pass