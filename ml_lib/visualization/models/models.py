"""
Modelos para componentes de visualización en ml_lib
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .enums import PlotStyle, ColorScheme, LineStyle, MarkerStyle, ImageFormat


@dataclass
class PlotConfig:
    """Configuración para componentes de visualización con tipos fuertemente tipados."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (10, 6)
    style: PlotStyle = PlotStyle.DEFAULT
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    alpha: float = 0.7
    grid: bool = True
    legend: bool = True
    color: Union[str, List[str]] = "blue"
    linestyle: LineStyle = LineStyle.SOLID
    linewidth: float = 2.0
    rotation: int = 45
    colorbar_label: str = "Value"
    center: Optional[float] = None
    # Solo para backward compatibility - preferir usar campos específicos
    _additional_params: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Validación de configuración."""
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if self.linewidth <= 0:
            raise ValueError(f"linewidth must be positive, got {self.linewidth}")
        if any(dim <= 0 for dim in self.figsize):
            raise ValueError(f"figsize dimensions must be positive, got {self.figsize}")

    @property
    def color_map(self) -> str:
        """Backward compatibility property."""
        return self.color_scheme.value

    @property
    def additional_params(self) -> Dict[str, Any]:
        """Backward compatibility property (deprecated)."""
        return self._additional_params


@dataclass
class VisualizationMetadata:
    """Metadatos para componentes de visualización con tipado fuerte."""

    plot_type: str  # Mantener como str para flexibilidad (puede ser PlotType.value)
    created_at: str = field(init=False)
    data_shape: Tuple[int, ...] = ()
    features_used: List[str] = field(default_factory=list)
    transformation_applied: str = ""
    execution_time_ms: Optional[float] = None
    # Métricas específicas en lugar de dict genérico
    num_data_points: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    # Solo para métricas adicionales no previstas
    _custom_metrics: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from datetime import datetime
        self.created_at = datetime.now().isoformat()

    def add_metric(self, name: str, value: float) -> None:
        """Añade una métrica personalizada."""
        self._custom_metrics[name] = value

    def get_metric(self, name: str) -> Optional[float]:
        """Obtiene una métrica por nombre."""
        # Intentar primero atributos específicos
        if name == "execution_time_ms":
            return self.execution_time_ms
        elif name == "num_data_points":
            return float(self.num_data_points) if self.num_data_points is not None else None
        elif name == "memory_usage_mb":
            return self.memory_usage_mb
        # Si no, buscar en custom metrics
        return self._custom_metrics.get(name)


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
    """Datos específicos para gráficos de líneas con tipado fuerte."""

    x: np.ndarray
    y: np.ndarray
    title: str = "Line Plot"
    xlabel: str = "X"
    ylabel: str = "Y"
    linewidth: float = 2.0
    linestyle: LineStyle = LineStyle.SOLID
    color: str = "blue"
    grid: bool = True

    def __post_init__(self):
        """Validación de datos."""
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have same length: {len(self.x)} != {len(self.y)}")
        if self.linewidth <= 0:
            raise ValueError(f"linewidth must be positive, got {self.linewidth}")


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
    """Datos específicos para heatmaps con tipado fuerte."""

    data: np.ndarray
    x_labels: Optional[List[str]] = None
    y_labels: Optional[List[str]] = None
    title: str = "Heatmap"
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    center: Optional[float] = None
    colorbar_label: str = "Value"
    rotation: int = 45

    def __post_init__(self):
        """Validación de datos."""
        if self.data.ndim != 2:
            raise ValueError(f"heatmap data must be 2D, got {self.data.ndim}D")

        if self.x_labels is not None and len(self.x_labels) != self.data.shape[1]:
            raise ValueError(
                f"x_labels length {len(self.x_labels)} must match data columns {self.data.shape[1]}"
            )

        if self.y_labels is not None and len(self.y_labels) != self.data.shape[0]:
            raise ValueError(
                f"y_labels length {len(self.y_labels)} must match data rows {self.data.shape[0]}"
            )

    @property
    def color_map(self) -> str:
        """Backward compatibility property."""
        return self.color_scheme.value


@dataclass
class BoxPlotData:
    """Datos específicos para gráficos de caja (boxplot)."""

    data: list[np.ndarray]  # Lista de arrays, uno por caja
    labels: Optional[List[str]] = None
    title: str = "Box Plot"
    xlabel: str = "Categories"
    ylabel: str = "Values"
    show_means: bool = True
    show_fliers: bool = True  # Outliers
    vert: bool = True  # Vertical boxes
    grid: bool = True

    def __post_init__(self):
        """Validación de datos."""
        if not self.data:
            raise ValueError("data must not be empty")
        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError(
                f"labels length {len(self.labels)} must match data length {len(self.data)}"
            )


@dataclass
class ViolinPlotData:
    """Datos específicos para gráficos de violín."""

    data: list[np.ndarray]  # Lista de arrays, uno por violín
    labels: Optional[List[str]] = None
    title: str = "Violin Plot"
    xlabel: str = "Categories"
    ylabel: str = "Values"
    show_means: bool = True
    show_extrema: bool = True
    show_median: bool = True
    vert: bool = True
    grid: bool = True

    def __post_init__(self):
        """Validación de datos."""
        if not self.data:
            raise ValueError("data must not be empty")
        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError(
                f"labels length {len(self.labels)} must match data length {len(self.data)}"
            )


@dataclass
class HistogramData:
    """Datos específicos para histogramas."""

    data: np.ndarray
    bins: int | np.ndarray | str = 30
    title: str = "Histogram"
    xlabel: str = "Values"
    ylabel: str = "Frequency"
    density: bool = False  # Si True, muestra densidad en lugar de conteos
    cumulative: bool = False
    alpha: float = 0.7
    color: str = "blue"
    grid: bool = True

    def __post_init__(self):
        """Validación de datos."""
        if self.data.size == 0:
            raise ValueError("data must not be empty")
        if isinstance(self.bins, int) and self.bins <= 0:
            raise ValueError(f"bins must be positive, got {self.bins}")
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")


@dataclass
class PiePlotData:
    """Datos específicos para gráficos de pastel (pie chart)."""

    values: np.ndarray
    labels: List[str]
    title: str = "Pie Chart"
    explode: Optional[np.ndarray] = None  # Para separar sectores
    autopct: str = "%1.1f%%"  # Formato de porcentaje
    startangle: float = 90
    colors: Optional[List[str]] = None
    shadow: bool = False

    def __post_init__(self):
        """Validación de datos."""
        if len(self.values) != len(self.labels):
            raise ValueError(
                f"values length {len(self.values)} must match labels length {len(self.labels)}"
            )
        if self.explode is not None and len(self.explode) != len(self.values):
            raise ValueError(
                f"explode length must match values length {len(self.values)}"
            )
        if np.any(self.values < 0):
            raise ValueError("values must be non-negative")


@dataclass
class ContourPlotData:
    """Datos específicos para gráficos de contorno."""

    X: np.ndarray  # Meshgrid X
    Y: np.ndarray  # Meshgrid Y
    Z: np.ndarray  # Valores
    title: str = "Contour Plot"
    xlabel: str = "X"
    ylabel: str = "Y"
    levels: int | np.ndarray = 10
    filled: bool = True  # Si True, contourf; si False, contour
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    colorbar: bool = True
    colorbar_label: str = "Values"

    def __post_init__(self):
        """Validación de datos."""
        if self.X.shape != self.Y.shape or self.X.shape != self.Z.shape:
            raise ValueError(
                f"X, Y, Z must have same shape. Got X:{self.X.shape}, Y:{self.Y.shape}, Z:{self.Z.shape}"
            )
        if isinstance(self.levels, int) and self.levels <= 0:
            raise ValueError(f"levels must be positive, got {self.levels}")
