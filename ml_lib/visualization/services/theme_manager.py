"""
Gestor de temas para visualización con soporte completo de matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional, Dict
import logging

from ..models.themes import Theme, AVAILABLE_THEMES


class ThemeManager:
    """Gestor centralizado de temas de visualización."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._current_theme: Optional[Theme] = None
        self._original_rcparams: Optional[Dict] = None

    @property
    def current_theme(self) -> Optional[Theme]:
        """Retorna el tema actual."""
        return self._current_theme

    @property
    def available_themes(self) -> list[str]:
        """Lista de temas disponibles."""
        return list(AVAILABLE_THEMES.keys())

    def apply_theme(self, theme_name: str) -> None:
        """
        Aplica un tema predefinido.

        Args:
            theme_name: Nombre del tema a aplicar

        Raises:
            ValueError: Si el tema no existe
        """
        if theme_name not in AVAILABLE_THEMES:
            available = ", ".join(self.available_themes)
            raise ValueError(
                f"Theme '{theme_name}' not found. Available themes: {available}"
            )

        theme = AVAILABLE_THEMES[theme_name]
        self._apply_theme_settings(theme)
        self._current_theme = theme
        self.logger.info(f"Applied theme: {theme.name}")

    def apply_custom_theme(self, theme: Theme) -> None:
        """
        Aplica un tema personalizado.

        Args:
            theme: Instancia de Theme personalizada
        """
        self._apply_theme_settings(theme)
        self._current_theme = theme
        self.logger.info(f"Applied custom theme: {theme.name}")

    def _apply_theme_settings(self, theme: Theme) -> None:
        """Aplica todas las configuraciones del tema a matplotlib."""
        # Guardar configuración original la primera vez
        if self._original_rcparams is None:
            self._original_rcparams = mpl.rcParams.copy()

        # Aplicar estilo base
        try:
            plt.style.use(theme.plot_style.value)
        except Exception as e:
            self.logger.warning(f"Could not apply plot style: {e}")

        # Configurar colores
        mpl.rcParams['figure.facecolor'] = theme.figure_facecolor
        mpl.rcParams['axes.facecolor'] = theme.axes_facecolor
        mpl.rcParams['axes.edgecolor'] = theme.palette.grid
        mpl.rcParams['axes.labelcolor'] = theme.palette.text
        mpl.rcParams['text.color'] = theme.palette.text
        mpl.rcParams['xtick.color'] = theme.palette.text_secondary
        mpl.rcParams['ytick.color'] = theme.palette.text_secondary
        mpl.rcParams['grid.color'] = theme.palette.grid
        mpl.rcParams['grid.alpha'] = theme.grid_alpha

        # Configurar líneas por defecto
        mpl.rcParams['lines.linewidth'] = theme.line_width
        mpl.rcParams['lines.linestyle'] = theme.line_style.value

        # Configurar fuentes
        mpl.rcParams['font.family'] = theme.font_family.split(',')[0].strip()
        mpl.rcParams['font.size'] = theme.font_size
        mpl.rcParams['axes.titlesize'] = theme.font_size + 2
        mpl.rcParams['axes.labelsize'] = theme.font_size
        mpl.rcParams['xtick.labelsize'] = theme.font_size - 1
        mpl.rcParams['ytick.labelsize'] = theme.font_size - 1
        mpl.rcParams['legend.fontsize'] = theme.font_size - 1

        # Configurar paleta de colores por defecto
        color_cycle = theme.palette.to_list()
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_cycle)

        # Configurar grid
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['axes.axisbelow'] = True

    def reset_theme(self) -> None:
        """Restaura la configuración original de matplotlib."""
        if self._original_rcparams is not None:
            mpl.rcParams.update(self._original_rcparams)
            self._current_theme = None
            self.logger.info("Theme reset to original matplotlib defaults")
        else:
            self.logger.warning("No original rcParams to restore")

    def get_theme_preview(self, theme_name: str) -> Dict[str, str]:
        """
        Obtiene un preview de los colores del tema.

        Args:
            theme_name: Nombre del tema

        Returns:
            Diccionario con los colores del tema

        Raises:
            ValueError: Si el tema no existe
        """
        if theme_name not in AVAILABLE_THEMES:
            raise ValueError(f"Theme '{theme_name}' not found")

        theme = AVAILABLE_THEMES[theme_name]
        palette = theme.palette

        return {
            "primary": palette.primary,
            "secondary": palette.secondary,
            "accent": palette.accent,
            "background": palette.background,
            "surface": palette.surface,
            "text": palette.text,
            "success": palette.success,
            "warning": palette.warning,
            "error": palette.error,
            "info": palette.info,
        }

    def create_figure_with_theme(
        self,
        theme_name: Optional[str] = None,
        figsize: tuple[int, int] = (10, 6)
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Crea una figura con el tema aplicado.

        Args:
            theme_name: Nombre del tema (usa el actual si es None)
            figsize: Tamaño de la figura

        Returns:
            Tupla (fig, ax)
        """
        if theme_name:
            self.apply_theme(theme_name)

        fig, ax = plt.subplots(figsize=figsize)

        if self._current_theme:
            fig.patch.set_facecolor(self._current_theme.figure_facecolor)
            ax.set_facecolor(self._current_theme.axes_facecolor)

        return fig, ax

    def get_color_palette(
        self,
        theme_name: Optional[str] = None,
        as_list: bool = False
    ) -> Dict[str, str] | list[str]:
        """
        Obtiene la paleta de colores de un tema.

        Args:
            theme_name: Nombre del tema (usa el actual si es None)
            as_list: Si True, retorna solo lista de colores principales

        Returns:
            Paleta de colores

        Raises:
            ValueError: Si no hay tema activo y theme_name es None
        """
        if theme_name:
            if theme_name not in AVAILABLE_THEMES:
                raise ValueError(f"Theme '{theme_name}' not found")
            theme = AVAILABLE_THEMES[theme_name]
        elif self._current_theme:
            theme = self._current_theme
        else:
            raise ValueError("No active theme and no theme_name provided")

        if as_list:
            return theme.palette.to_list()
        else:
            return {
                "primary": theme.palette.primary,
                "secondary": theme.palette.secondary,
                "accent": theme.palette.accent,
                "success": theme.palette.success,
                "warning": theme.palette.warning,
                "error": theme.palette.error,
                "info": theme.palette.info,
            }


# Instancia global singleton
_global_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Obtiene la instancia global del ThemeManager."""
    global _global_theme_manager
    if _global_theme_manager is None:
        _global_theme_manager = ThemeManager()
    return _global_theme_manager


def apply_theme(theme_name: str) -> None:
    """
    Función de conveniencia para aplicar un tema globalmente.

    Args:
        theme_name: Nombre del tema a aplicar
    """
    manager = get_theme_manager()
    manager.apply_theme(theme_name)


def reset_theme() -> None:
    """Función de conveniencia para resetear el tema global."""
    manager = get_theme_manager()
    manager.reset_theme()


def list_themes() -> list[str]:
    """Función de conveniencia para listar temas disponibles."""
    manager = get_theme_manager()
    return manager.available_themes
