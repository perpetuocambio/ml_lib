"""
Interfaz base para todos los estimadores con tipado estricto.
"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Protocol
import numpy as np


X = TypeVar('X', bound=np.ndarray)
Y = TypeVar('Y', bound=np.ndarray)


class EstimatorInterface(ABC, Generic[X, Y]):
    """Interface base para todos los estimadores."""
    
    @abstractmethod
    def fit(self, X: X, y: Y, **kwargs) -> 'EstimatorInterface[X, Y]':
        """Entrena el modelo con los datos proporcionados."""
        pass
    
    @abstractmethod
    def predict(self, X: X) -> Y:
        """Realiza predicciones sobre nuevos datos."""
        pass
    
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Obtiene los hiperparámetros del modelo."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'EstimatorInterface[X, Y]':
        """Establece los hiperparámetros del modelo."""
        pass


class SupervisedEstimatorInterface(EstimatorInterface[X, Y], ABC):
    """Interface para estimadores supervisados."""
    pass


class UnsupervisedEstimatorInterface(EstimatorInterface[X, X], ABC):
    """Interface para estimadores no supervisados (clustering, etc.)."""
    
    @abstractmethod
    def fit_predict(self, X: X) -> X:
        """Ajusta el modelo y devuelve las predicciones."""
        pass