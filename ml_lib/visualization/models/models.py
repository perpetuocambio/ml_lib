"""
Modelos para componentes de visualización en ml_lib
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class PlotConfig:
    """Configuración para componentes de visualización."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (10, 6)
    style: str = "default"
    color_map: str = "viridis"
    alpha: float = 0.7
    grid: bool = True
    legend: bool = True
    color: Union[str, List[str]] = "blue"
    linestyle: str = "-"
    linewidth: float = 2.0
    rotation: int = 45
    colorbar_label: str = "Value"
    center: Optional[float] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationMetadata:
    """Metadatos para componentes de visualización."""

    plot_type: str
    created_at: str = ""
    data_shape: Tuple[int, ...] = ()
    features_used: List[str] = field(default_factory=list)
    transformation_applied: str = ""
    custom_params: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        from datetime import datetime

        self.created_at = datetime.now().isoformat()


@dataclass
class ScatterPlotData:
    """Datos específicos para gráficos de dispersión."""

    x: np.ndarray
    y: np.ndarray
    labels: Optional[List[str]] = None
    colors: Optional[np.ndarray] = None
    sizes: Optional[np.ndarray] = None
    alpha: float = 0.7
    title: str = "Scatter Plot"
    xlabel: str = "X"
    ylabel: str = "Y"
    grid: bool = True
    legend: bool = True


@dataclass
class LinePlotData:
    """Datos específicos para gráficos de líneas."""

    x: np.ndarray
    y: np.ndarray
    title: str = "Line Plot"
    xlabel: str = "X"
    ylabel: str = "Y"
    linewidth: float = 2.0
    linestyle: str = "-"
    color: str = "blue"
    grid: bool = True


@dataclass
class BarPlotData:
    """Datos específicos para gráficos de barras."""

    x: np.ndarray  # Puede ser índices o categorías
    heights: np.ndarray
    labels: Optional[List[str]] = None
    title: str = "Bar Plot"
    xlabel: str = "Categories"
    ylabel: str = "Values"
    color: str = "blue"
    alpha: float = 0.7
    rotation: int = 45
    grid: bool = True


@dataclass
class HeatmapData:
    """Datos específicos para heatmaps."""

    data: np.ndarray
    x_labels: Optional[List[str]] = None
    y_labels: Optional[List[str]] = None
    title: str = "Heatmap"
    color_map: str = "viridis"
    center: Optional[float] = None
    colorbar_label: str = "Value"
    rotation: int = 45
