"""
Servicios para componentes de visualización en ml_lib
"""
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List
import numpy as np
import logging


class VisualizationService:
    """Servicio para operaciones generales de visualización."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def setup_plot_style(self, style: str = "seaborn"):
        """Configura el estilo de los gráficos."""
        if style == "seaborn":
            sns.set_style("whitegrid")
        elif style == "matplotlib":
            plt.style.use('default')
        else:
            plt.style.use(style)
        
        self.logger.info(f"Estilo de gráficos configurado a: {style}")
    
    def validate_plot_data(self, data: np.ndarray) -> bool:
        """Valida que los datos sean adecuados para visualización."""
        if data.size == 0:
            self.logger.error("Datos vacíos para visualización")
            return False
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            self.logger.warning("Datos contienen NaN o infinitos")
            # Podríamos imputar o manejar estos valores
        
        return True
    
    def create_figure(self, figsize: tuple = (10, 6)) -> plt.Figure:
        """Crea una figura matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax


class PlottingService:
    """Servicio para crear tipos específicos de gráficos."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def create_scatter_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: List[str] = None,
        colors: np.ndarray = None,
        **kwargs
    ) -> plt.Figure:
        """Crea un gráfico de dispersión."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        scatter = ax.scatter(x, y, c=colors, alpha=kwargs.get('alpha', 0.7))
        
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Scatter Plot'))
        
        if kwargs.get('grid', True):
            ax.grid(True, alpha=0.3)
        
        if kwargs.get('legend', True) and labels:
            ax.legend()
        
        return fig
    
    def create_line_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> plt.Figure:
        """Crea un gráfico de líneas."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        ax.plot(x, y, linewidth=kwargs.get('linewidth', 2), 
                linestyle=kwargs.get('linestyle', '-'))
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Line Plot'))
        
        if kwargs.get('grid', True):
            ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_bar_plot(
        self,
        x: np.ndarray,
        heights: np.ndarray,
        labels: List[str] = None,
        **kwargs
    ) -> plt.Figure:
        """Crea un gráfico de barras."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        bars = ax.bar(x if len(x) == len(heights) else range(len(heights)), 
                     heights, 
                     color=kwargs.get('color', 'blue'),
                     alpha=kwargs.get('alpha', 0.7))
        
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=kwargs.get('rotation', 45), ha="right")
        
        ax.set_xlabel(kwargs.get('xlabel', 'Categories'))
        ax.set_ylabel(kwargs.get('ylabel', 'Values'))
        ax.set_title(kwargs.get('title', 'Bar Plot'))
        
        if kwargs.get('grid', True):
            ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str] = None,
        y_labels: List[str] = None,
        **kwargs
    ) -> plt.Figure:
        """Crea un heatmap."""
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        heatmap = sns.heatmap(
            data,
            ax=ax,
            cmap=kwargs.get('color_map', 'viridis'),
            center=kwargs.get('center', 0),
            cbar_kws={'label': kwargs.get('colorbar_label', 'Value')}
        )
        
        if x_labels:
            ax.set_xticklabels(x_labels, rotation=kwargs.get('rotation', 45), ha="right")
        if y_labels:
            ax.set_yticklabels(y_labels, rotation=0)
        
        ax.set_title(kwargs.get('title', 'Heatmap'))
        
        return fig