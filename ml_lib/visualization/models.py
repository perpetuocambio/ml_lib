"""
Modelos para componentes de visualización en ml_lib
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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

    def __post_init__(self):
        from datetime import datetime
        self.created_at = datetime.now().isoformat()