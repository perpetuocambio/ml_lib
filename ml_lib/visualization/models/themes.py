"""
Temas predefinidos elegantes para visualización.
"""

from dataclasses import dataclass
from typing import Dict, List
from .enums import PlotStyle, ColorScheme, LineStyle


@dataclass
class ColorPalette:
    """Paleta de colores para un tema."""

    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text: str
    text_secondary: str
    grid: str

    # Colores adicionales
    success: str = "#4CAF50"
    warning: str = "#FF9800"
    error: str = "#F44336"
    info: str = "#2196F3"

    def to_list(self) -> List[str]:
        """Retorna los colores principales como lista."""
        return [self.primary, self.secondary, self.accent]


@dataclass
class Theme:
    """Tema completo de visualización."""

    name: str
    palette: ColorPalette
    plot_style: PlotStyle
    color_scheme: ColorScheme
    font_family: str = "sans-serif"
    font_size: int = 10
    line_width: float = 2.0
    line_style: LineStyle = LineStyle.SOLID
    grid_alpha: float = 0.3
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"


# ============================================================================
# TEMAS PREDEFINIDOS
# ============================================================================

# Material Design Theme
MATERIAL_PALETTE = ColorPalette(
    primary="#6200EE",      # Deep Purple
    secondary="#03DAC6",    # Teal
    accent="#BB86FC",       # Light Purple
    background="#FFFFFF",
    surface="#F5F5F5",
    text="#000000",
    text_secondary="#757575",
    grid="#E0E0E0",
)

MATERIAL_THEME = Theme(
    name="Material Design",
    palette=MATERIAL_PALETTE,
    plot_style=PlotStyle.SEABORN_WHITEGRID,
    color_scheme=ColorScheme.VIRIDIS,
    font_family="Roboto, sans-serif",
    font_size=11,
    line_width=2.5,
    grid_alpha=0.25,
)

# Material Dark Theme
MATERIAL_DARK_PALETTE = ColorPalette(
    primary="#BB86FC",      # Light Purple
    secondary="#03DAC6",    # Teal
    accent="#CF6679",       # Pink
    background="#121212",
    surface="#1E1E1E",
    text="#FFFFFF",
    text_secondary="#B0B0B0",
    grid="#2C2C2C",
)

MATERIAL_DARK_THEME = Theme(
    name="Material Dark",
    palette=MATERIAL_DARK_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.PLASMA,
    font_family="Roboto, sans-serif",
    font_size=11,
    line_width=2.5,
    figure_facecolor="#121212",
    axes_facecolor="#1E1E1E",
    grid_alpha=0.2,
)

# Nord Theme
NORD_PALETTE = ColorPalette(
    primary="#5E81AC",      # Frost Blue
    secondary="#88C0D0",    # Frost Light Blue
    accent="#BF616A",       # Aurora Red
    background="#2E3440",   # Polar Night
    surface="#3B4252",
    text="#ECEFF4",         # Snow Storm
    text_secondary="#D8DEE9",
    grid="#434C5E",
    success="#A3BE8C",
    warning="#EBCB8B",
    error="#BF616A",
    info="#81A1C1",
)

NORD_THEME = Theme(
    name="Nord",
    palette=NORD_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.CREST,
    font_family="Source Code Pro, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#2E3440",
    axes_facecolor="#3B4252",
    grid_alpha=0.3,
)

# Solarized Light Theme
SOLARIZED_LIGHT_PALETTE = ColorPalette(
    primary="#268BD2",      # Blue
    secondary="#2AA198",    # Cyan
    accent="#D33682",       # Magenta
    background="#FDF6E3",   # Base3
    surface="#EEE8D5",      # Base2
    text="#657B83",         # Base00
    text_secondary="#839496",
    grid="#93A1A1",
    success="#859900",
    warning="#B58900",
    error="#DC322F",
    info="#268BD2",
)

SOLARIZED_LIGHT_THEME = Theme(
    name="Solarized Light",
    palette=SOLARIZED_LIGHT_PALETTE,
    plot_style=PlotStyle.SEABORN_WHITEGRID,
    color_scheme=ColorScheme.COOLWARM,
    font_family="Fira Code, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#FDF6E3",
    axes_facecolor="#EEE8D5",
    grid_alpha=0.3,
)

# Solarized Dark Theme
SOLARIZED_DARK_PALETTE = ColorPalette(
    primary="#268BD2",      # Blue
    secondary="#2AA198",    # Cyan
    accent="#D33682",       # Magenta
    background="#002B36",   # Base03
    surface="#073642",      # Base02
    text="#839496",         # Base0
    text_secondary="#586E75",
    grid="#586E75",
    success="#859900",
    warning="#B58900",
    error="#DC322F",
    info="#268BD2",
)

SOLARIZED_DARK_THEME = Theme(
    name="Solarized Dark",
    palette=SOLARIZED_DARK_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.ROCKET,
    font_family="Fira Code, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#002B36",
    axes_facecolor="#073642",
    grid_alpha=0.3,
)

# Dracula Theme
DRACULA_PALETTE = ColorPalette(
    primary="#BD93F9",      # Purple
    secondary="#8BE9FD",    # Cyan
    accent="#FF79C6",       # Pink
    background="#282A36",
    surface="#44475A",
    text="#F8F8F2",
    text_secondary="#6272A4",
    grid="#44475A",
    success="#50FA7B",
    warning="#F1FA8C",
    error="#FF5555",
    info="#8BE9FD",
)

DRACULA_THEME = Theme(
    name="Dracula",
    palette=DRACULA_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.MAKO,
    font_family="Fira Code, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#282A36",
    axes_facecolor="#44475A",
    grid_alpha=0.25,
)

# Monokai Theme
MONOKAI_PALETTE = ColorPalette(
    primary="#66D9EF",      # Cyan
    secondary="#A6E22E",    # Green
    accent="#F92672",       # Pink
    background="#272822",
    surface="#3E3D32",
    text="#F8F8F2",
    text_secondary="#75715E",
    grid="#49483E",
    success="#A6E22E",
    warning="#E6DB74",
    error="#F92672",
    info="#66D9EF",
)

MONOKAI_THEME = Theme(
    name="Monokai",
    palette=MONOKAI_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.INFERNO,
    font_family="Fira Code, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#272822",
    axes_facecolor="#3E3D32",
    grid_alpha=0.2,
)

# One Dark Theme
ONE_DARK_PALETTE = ColorPalette(
    primary="#61AFEF",      # Blue
    secondary="#56B6C2",    # Cyan
    accent="#C678DD",       # Purple
    background="#282C34",
    surface="#21252B",
    text="#ABB2BF",
    text_secondary="#5C6370",
    grid="#3E4451",
    success="#98C379",
    warning="#E5C07B",
    error="#E06C75",
    info="#61AFEF",
)

ONE_DARK_THEME = Theme(
    name="One Dark",
    palette=ONE_DARK_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.FLARE,
    font_family="Fira Code, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#282C34",
    axes_facecolor="#21252B",
    grid_alpha=0.25,
)

# Gruvbox Theme
GRUVBOX_PALETTE = ColorPalette(
    primary="#83A598",      # Blue
    secondary="#8EC07C",    # Aqua
    accent="#D3869B",       # Purple
    background="#282828",
    surface="#3C3836",
    text="#EBDBB2",
    text_secondary="#A89984",
    grid="#504945",
    success="#B8BB26",
    warning="#FABD2F",
    error="#FB4934",
    info="#83A598",
)

GRUVBOX_THEME = Theme(
    name="Gruvbox",
    palette=GRUVBOX_PALETTE,
    plot_style=PlotStyle.DARK_BACKGROUND,
    color_scheme=ColorScheme.RD_YL_GN,
    font_family="JetBrains Mono, monospace",
    font_size=10,
    line_width=2.0,
    figure_facecolor="#282828",
    axes_facecolor="#3C3836",
    grid_alpha=0.3,
)

# Scientific (Publication Ready) Theme
SCIENTIFIC_PALETTE = ColorPalette(
    primary="#1F77B4",      # Tab Blue
    secondary="#FF7F0E",    # Tab Orange
    accent="#2CA02C",       # Tab Green
    background="#FFFFFF",
    surface="#F9F9F9",
    text="#000000",
    text_secondary="#333333",
    grid="#CCCCCC",
)

SCIENTIFIC_THEME = Theme(
    name="Scientific",
    palette=SCIENTIFIC_PALETTE,
    plot_style=PlotStyle.CLASSIC,
    color_scheme=ColorScheme.TAB10,
    font_family="Times New Roman, serif",
    font_size=12,
    line_width=1.5,
    grid_alpha=0.5,
)

# Minimal Theme
MINIMAL_PALETTE = ColorPalette(
    primary="#000000",
    secondary="#666666",
    accent="#999999",
    background="#FFFFFF",
    surface="#FAFAFA",
    text="#000000",
    text_secondary="#666666",
    grid="#E5E5E5",
)

MINIMAL_THEME = Theme(
    name="Minimal",
    palette=MINIMAL_PALETTE,
    plot_style=PlotStyle.DEFAULT,
    color_scheme=ColorScheme.VIRIDIS,
    font_family="Helvetica, Arial, sans-serif",
    font_size=10,
    line_width=1.5,
    grid_alpha=0.2,
)


# Registro de todos los temas
AVAILABLE_THEMES: Dict[str, Theme] = {
    "material": MATERIAL_THEME,
    "material_dark": MATERIAL_DARK_THEME,
    "nord": NORD_THEME,
    "solarized_light": SOLARIZED_LIGHT_THEME,
    "solarized_dark": SOLARIZED_DARK_THEME,
    "dracula": DRACULA_THEME,
    "monokai": MONOKAI_THEME,
    "one_dark": ONE_DARK_THEME,
    "gruvbox": GRUVBOX_THEME,
    "scientific": SCIENTIFIC_THEME,
    "minimal": MINIMAL_THEME,
}
