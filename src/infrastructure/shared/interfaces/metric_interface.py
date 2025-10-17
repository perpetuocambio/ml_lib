"""
Interfaz para métricas de evaluación con tipado estricto.
"""

from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np


Y = TypeVar("Y", bound=np.ndarray)


class MetricInterface(ABC):
    """Interface base para métricas de evaluación."""

    @abstractmethod
    def evaluate(self, y_true: Y, y_pred: Y) -> float:
        """Evalúa la métrica dados los valores verdaderos y predichos."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Obtiene el nombre de la métrica."""
        pass


class SupervisedMetricInterface(MetricInterface, ABC):
    """Interface para métricas de tareas supervisadas."""

    pass


class UnsupervisedMetricInterface(MetricInterface, ABC):
    """Interface para métricas de tareas no supervisadas (ej. clustering)."""

    @abstractmethod
    def evaluate_unsupervised(self, X: np.ndarray, labels: Y) -> float:
        """Evalúa métrica no supervisada usando datos y etiquetas de cluster."""
        pass
