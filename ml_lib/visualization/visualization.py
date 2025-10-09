"""
Implementación concreta de componentes de visualización para ml_lib
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from .interfaces import VisualizationInterface, PlotTypeInterface
from .models import PlotConfig, VisualizationMetadata
from .services import VisualizationService, PlottingService
from .handlers import VisualizationErrorHandler, VisualizationConfigHandler, ImageExportHandler
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
    
    @VisualizationErrorHandler.handle_visualization_error
    def plot_scatter(self, x: np.ndarray, y: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de dispersión."""
        # Combinar configuración por defecto con parámetros específicos
        plot_params = {
            'figsize': kwargs.get('figsize', self.config.figsize),
            'xlabel': kwargs.get('xlabel', self.config.xlabel),
            'ylabel': kwargs.get('ylabel', self.config.ylabel),
            'title': kwargs.get('title', self.config.title),
            'alpha': kwargs.get('alpha', self.config.alpha),
            'grid': kwargs.get('grid', self.config.grid),
            'legend': kwargs.get('legend', self.config.legend),
        }
        
        # Validar datos
        self.visualization_service.validate_plot_data(x)
        self.visualization_service.validate_plot_data(y)
        
        # Sanitizar parámetros
        plot_params = self.config_handler.sanitize_plot_params(plot_params)
        
        return self.plotting_service.create_scatter_plot(x, y, **plot_params)
    
    @VisualizationErrorHandler.handle_visualization_error
    def plot_line(self, x: np.ndarray, y: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de líneas."""
        plot_params = {
            'figsize': kwargs.get('figsize', self.config.figsize),
            'xlabel': kwargs.get('xlabel', self.config.xlabel),
            'ylabel': kwargs.get('ylabel', self.config.ylabel),
            'title': kwargs.get('title', self.config.title),
            'linewidth': kwargs.get('linewidth', 2),
            'linestyle': kwargs.get('linestyle', '-'),
            'grid': kwargs.get('grid', self.config.grid),
        }
        
        # Validar y sanitizar
        self.visualization_service.validate_plot_data(x)
        self.visualization_service.validate_plot_data(y)
        plot_params = self.config_handler.sanitize_plot_params(plot_params)
        
        return self.plotting_service.create_line_plot(x, y, **plot_params)
    
    @VisualizationErrorHandler.handle_visualization_error
    def plot_bar(self, x: np.ndarray, heights: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un gráfico de barras."""
        plot_params = {
            'figsize': kwargs.get('figsize', self.config.figsize),
            'xlabel': kwargs.get('xlabel', self.config.xlabel),
            'ylabel': kwargs.get('ylabel', self.config.ylabel),
            'title': kwargs.get('title', self.config.title),
            'color': kwargs.get('color', 'blue'),
            'alpha': kwargs.get('alpha', self.config.alpha),
            'rotation': kwargs.get('rotation', 45),
            'grid': kwargs.get('grid', self.config.grid),
        }
        
        # Validar y sanitizar
        self.visualization_service.validate_plot_data(heights)
        plot_params = self.config_handler.sanitize_plot_params(plot_params)
        
        return self.plotting_service.create_bar_plot(x, heights, **plot_params)
    
    @VisualizationErrorHandler.handle_visualization_error
    def plot_heatmap(self, data: np.ndarray, **kwargs) -> plt.Figure:
        """Crea un heatmap."""
        plot_params = {
            'figsize': kwargs.get('figsize', self.config.figsize),
            'title': kwargs.get('title', self.config.title),
            'color_map': kwargs.get('color_map', self.config.color_map),
            'colorbar_label': kwargs.get('colorbar_label', 'Value'),
            'rotation': kwargs.get('rotation', 45),
        }
        
        # Validar y sanitizar
        self.visualization_service.validate_plot_data(data)
        plot_params = self.config_handler.sanitize_plot_params(plot_params)
        
        return self.plotting_service.create_heatmap(data, **plot_params)


class VisualizationFactory:
    """Fábrica para crear componentes de visualización."""
    
    @staticmethod
    def create_visualization(config: Optional[PlotConfig] = None) -> GeneralVisualization:
        """Crea una instancia de visualización con la configuración especificada."""
        return GeneralVisualization(config)