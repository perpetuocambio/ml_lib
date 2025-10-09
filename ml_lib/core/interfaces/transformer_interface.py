"""
Interfaz para transformadores de datos con tipado estricto.
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union
import numpy as np


X = TypeVar('X', bound=np.ndarray)
Y = TypeVar('Y', bound=np.ndarray)


class TransformerInterface(ABC, Generic[X]):
    """Interface para transformadores de datos."""
    
    @abstractmethod
    def fit(self, X: X, y: Union[Y, None] = None) -> 'TransformerInterface[X]':
        """Aprende los parámetros de transformación."""
        pass
    
    @abstractmethod
    def transform(self, X: X) -> X:
        """Aplica la transformación a los datos."""
        pass
    
    def fit_transform(self, X: X, y: Union[Y, None] = None) -> X:
        """Ajusta y transforma en un solo paso."""
        return self.fit(X, y).transform(X)

    @abstractmethod
    def inverse_transform(self, X: X) -> X:
        """Aplica la transformación inversa."""
        pass