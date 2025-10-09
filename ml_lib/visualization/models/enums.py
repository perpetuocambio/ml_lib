"""
Enumeraciones para el módulo de visualización de ml_lib.
"""

from enum import Enum


class PlotType(Enum):
    """Tipos de gráficos disponibles."""
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    SURFACE = "surface"
    PIE = "pie"
    AREA = "area"
    SCATTER_3D = "scatter_3d"


class PlotStyle(Enum):
    """Estilos de visualización."""
    DEFAULT = "default"
    SEABORN = "seaborn"
    SEABORN_DARK = "seaborn-dark"
    SEABORN_DARKGRID = "seaborn-darkgrid"
    SEABORN_WHITEGRID = "seaborn-whitegrid"
    GGPLOT = "ggplot"
    DARK_BACKGROUND = "dark_background"
    BMH = "bmh"
    CLASSIC = "classic"
    FIVETHIRTYEIGHT = "fivethirtyeight"
    GRAYSCALE = "grayscale"


class ColorScheme(Enum):
    """Esquemas de color para gráficos."""
    # Sequential
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"

    # Diverging
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"
    RD_BU = "RdBu"
    RD_YL_GN = "RdYlGn"
    SPECTRAL = "Spectral"

    # Qualitative
    SET1 = "Set1"
    SET2 = "Set2"
    SET3 = "Set3"
    PAIRED = "Paired"
    TAB10 = "tab10"
    TAB20 = "tab20"

    # Perceptually Uniform
    ROCKET = "rocket"
    MAKO = "mako"
    FLARE = "flare"
    CREST = "crest"


class LineStyle(Enum):
    """Estilos de línea."""
    SOLID = "-"
    DASHED = "--"
    DASHDOT = "-."
    DOTTED = ":"
    NONE = ""


class MarkerStyle(Enum):
    """Estilos de marcador para scatter plots."""
    POINT = "."
    PIXEL = ","
    CIRCLE = "o"
    TRIANGLE_DOWN = "v"
    TRIANGLE_UP = "^"
    TRIANGLE_LEFT = "<"
    TRIANGLE_RIGHT = ">"
    SQUARE = "s"
    PENTAGON = "p"
    STAR = "*"
    HEXAGON = "h"
    PLUS = "+"
    X = "x"
    DIAMOND = "D"
    THIN_DIAMOND = "d"


class ImageFormat(Enum):
    """Formatos de imagen para exportación."""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    SVG = "svg"
    PDF = "pdf"
    EPS = "eps"
    TIFF = "tiff"
