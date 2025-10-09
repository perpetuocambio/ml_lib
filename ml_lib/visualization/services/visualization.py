"""
Implementación concreta de componentes de visualización para ml_lib
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .interfaces import VisualizationInterface, PlotTypeInterface
from .models import (
    PlotConfig,
)
from .services import VisualizationService, PlottingService
from .handlers import (
    VisualizationErrorHandler,
    VisualizationConfigHandler,
    ImageExportHandler,
)
from ml_lib.core import LoggingService


class GeneralVisualization(VisualizationInterface, PlotTypeInterface):
    """Implementación concreta de componentes de visualización generales."""

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.logger_service = LoggingService("GeneralVisualization")
        self.logger = self.logger_service.get_logger()
        self.visualization_service = VisualizationService(self.logger)
        self.plotting_service = PlottingService(self.logger)
        self.error_handler = VisualizationErrorHandler(self.logger)
        self.config_handler = VisualizationConfigHandler()
        self.export_handler = ImageExportHandler(self.logger)

        # Configurar estilo inicial
        self.visualization_service.setup_plot_style(self.config.style)

    def create_plot(self, data: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico general con los datos proporcionados."""
        return self.plot_scatter(data[:, 0], data[:, 1], **kwargs)

    def save_plot(self, figure: plt.Figure, filepath: str) -> None:
        """Guarda el gráfico en un archivo."""
        self.export_handler.save_figure(figure, filepath)

    def plot_scatter(self, x: np.ndarray, y: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de dispersión usando modelos de datos."""
        # Validar datos
        self.visualization_service.validate_plot_data(x)
        self.visualization_service.validate_plot_data(y)

        # Combinar configuración por defecto con parámetros específicos
        config = PlotConfig(**{k: v for k, v in self.config.__dict__.items()})
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return self.plotting_service.create_scatter_plot_from_config(
            x, y, config, labels=kwargs.get("labels"), colors=kwargs.get("colors")
        )

    def plot_line(self, x: np.ndarray, y: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de líneas usando modelos de datos."""
        # Validar datos
        self.visualization_service.validate_plot_data(x)
        self.visualization_service.validate_plot_data(y)

        # Combinar configuración por defecto con parámetros específicos
        config = PlotConfig(**{k: v for k, v in self.config.__dict__.items()})
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return self.plotting_service.create_line_plot_from_config(x, y, config)

    def plot_bar(self, x: np.ndarray, heights: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de barras usando modelos de datos."""
        # Validar datos
        self.visualization_service.validate_plot_data(heights)

        # Combinar configuración por defecto con parámetros específicos
        config = PlotConfig(**{k: v for k, v in self.config.__dict__.items()})
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return self.plotting_service.create_bar_plot_from_config(
            x, heights, config, labels=kwargs.get("labels")
        )

    def plot_heatmap(self, data: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un heatmap usando modelos de datos."""
        # Validar datos
        self.visualization_service.validate_plot_data(data)

        # Combinar configuración por defecto con parámetros específicos
        config = PlotConfig(**{k: v for k, v in self.config.__dict__.items()})
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return self.plotting_service.create_heatmap_from_config(
            data,
            config,
            x_labels=kwargs.get("x_labels"),
            y_labels=kwargs.get("y_labels"),
        )


class VisualizationFactory:
    """Fábrica para crear componentes de visualización."""

    @staticmethod
    def create_visualization(
        config: Optional[PlotConfig] = None,
    ) -> GeneralVisualization:
        """Crea una instancia de visualización con la configuración especificada."""
        return GeneralVisualization(config)
