"""
Interfaces para componentes de visualización en ml_lib
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class VisualizationInterface(ABC):
    """Interfaz base para componentes de visualización."""

    @abstractmethod
    def create_plot(self, data: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico con los datos proporcionados."""
        pass

    @abstractmethod
    def save_plot(self, figure: plt.Figure, filepath: str) -> None:
        """Guarda el gráfico en un archivo."""
        pass


class PlotTypeInterface(ABC):
    """Interfaz para tipos específicos de gráficos."""

    @abstractmethod
    def plot_scatter(self, x: np.ndarray, y: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de dispersión."""
        pass

    @abstractmethod
    def plot_line(self, x: np.ndarray, y: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de líneas."""
        pass

    @abstractmethod
    def plot_bar(self, x: np.ndarray, heights: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de barras."""
        pass

    @abstractmethod
    def plot_heatmap(self, data: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un heatmap."""
        pass
