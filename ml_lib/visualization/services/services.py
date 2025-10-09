"""
Servicios para componentes de visualización en ml_lib
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import numpy as np
import logging

# Importar los modelos específicos
from ..models import (
    PlotConfig,
    ScatterPlotData,
    LinePlotData,
    BarPlotData,
    HeatmapData,
    BoxPlotData,
    ViolinPlotData,
    HistogramData,
    PiePlotData,
    ContourPlotData,
)


class VisualizationService:
    """Servicio para operaciones generales de visualización."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def setup_plot_style(self, style: str = "seaborn"):
        """Configura el estilo de los gráficos."""
        if style == "seaborn":
            sns.set_style("whitegrid")
        elif style == "matplotlib":
            plt.style.use("default")
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

    def create_scatter_plot(self, data: ScatterPlotData) -> plt.Figure:
        """Crea un gráfico de dispersión usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(data.x.size, 6) if data.x.size > 6 else (10, 6))

        scatter = ax.scatter(data.x, data.y, c=data.colors, alpha=data.alpha)

        if data.labels:
            for i, label in enumerate(data.labels):
                ax.annotate(
                    label,
                    (data.x[i], data.y[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        if data.grid:
            ax.grid(True, alpha=0.3)

        if data.legend and data.labels:
            ax.legend()

        return fig

    def create_line_plot(self, data: LinePlotData) -> plt.Figure:
        """Crea un gráfico de líneas usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(data.x.size, 6) if data.x.size > 6 else (10, 6))

        ax.plot(
            data.x,
            data.y,
            linewidth=data.linewidth,
            linestyle=data.linestyle,
            color=data.color,
        )

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        if data.grid:
            ax.grid(True, alpha=0.3)

        return fig

    def create_bar_plot(self, data: BarPlotData) -> plt.Figure:
        """Crea un gráfico de barras usando modelo de datos."""
        fig, ax = plt.subplots(
            figsize=(len(data.heights), 6) if len(data.heights) > 6 else (10, 6)
        )

        x_positions = (
            data.x if len(data.x) == len(data.heights) else np.arange(len(data.heights))
        )
        bars = ax.bar(x_positions, data.heights, color=data.color, alpha=data.alpha)

        if data.labels:
            ax.set_xticks(range(len(data.labels)))
            ax.set_xticklabels(data.labels, rotation=data.rotation, ha="right")

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        if data.grid:
            ax.grid(True, alpha=0.3)

        return fig

    def create_heatmap(self, data: HeatmapData) -> plt.Figure:
        """Crea un heatmap usando modelo de datos."""
        import seaborn as sns

        fig, ax = plt.subplots(
            figsize=(
                len(data.data[0]) if len(data.data) > 0 else 10,
                len(data.data) if len(data.data) > 0 else 8,
            )
        )

        heatmap = sns.heatmap(
            data.data,
            ax=ax,
            cmap=data.color_map,
            center=data.center,
            cbar_kws={"label": data.colorbar_label},
        )

        if data.x_labels:
            ax.set_xticklabels(data.x_labels, rotation=data.rotation, ha="right")
        if data.y_labels:
            ax.set_yticklabels(data.y_labels, rotation=0)

        ax.set_title(data.title)

        return fig

    def create_scatter_plot_from_config(
        self,
        x: np.ndarray,
        y: np.ndarray,
        config: PlotConfig,
        labels: Optional[List[str]] = None,
        colors: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """Crea un gráfico de dispersión desde una configuración."""
        data = ScatterPlotData(
            x=x,
            y=y,
            labels=labels,
            colors=colors,
            alpha=config.alpha,
            title=config.title,
            xlabel=config.xlabel,
            ylabel=config.ylabel,
            grid=config.grid,
            legend=config.legend,
        )
        return self.create_scatter_plot(data)

    def create_line_plot_from_config(
        self, x: np.ndarray, y: np.ndarray, config: PlotConfig
    ) -> plt.Figure:
        """Crea un gráfico de líneas desde una configuración."""
        data = LinePlotData(
            x=x,
            y=y,
            title=config.title,
            xlabel=config.xlabel,
            ylabel=config.ylabel,
            linewidth=config.linewidth,
            linestyle=config.linestyle,
            color=config.color,
            grid=config.grid,
        )
        return self.create_line_plot(data)

    def create_bar_plot_from_config(
        self,
        x: np.ndarray,
        heights: np.ndarray,
        config: PlotConfig,
        labels: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Crea un gráfico de barras desde una configuración."""
        data = BarPlotData(
            x=x,
            heights=heights,
            labels=labels,
            title=config.title,
            xlabel=config.xlabel,
            ylabel=config.ylabel,
            color=config.color,
            alpha=config.alpha,
            rotation=config.rotation,
        )
        return self.create_bar_plot(data)

    def create_heatmap_from_config(
        self,
        data_matrix: np.ndarray,
        config: PlotConfig,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Crea un heatmap desde una configuración."""
        data = HeatmapData(
            data=data_matrix,
            x_labels=x_labels,
            y_labels=y_labels,
            title=config.title,
            color_map=config.color_map,
            center=config.center,
            colorbar_label=config.colorbar_label,
            rotation=config.rotation,
        )
        return self.create_heatmap(data)
    def create_box_plot(self, data: BoxPlotData) -> plt.Figure:
        """Crea un gráfico de caja (boxplot) usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(
            data.data,
            labels=data.labels,
            showmeans=data.show_means,
            showfliers=data.show_fliers,
            vert=data.vert,
            patch_artist=True,
        )

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        if data.grid:
            ax.grid(True, alpha=0.3, axis='y' if data.vert else 'x')

        return fig

    def create_violin_plot(self, data: ViolinPlotData) -> plt.Figure:
        """Crea un gráfico de violín usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(10, 6))

        parts = ax.violinplot(
            data.data,
            showmeans=data.show_means,
            showextrema=data.show_extrema,
            showmedians=data.show_median,
            vert=data.vert,
        )

        if data.labels:
            ax.set_xticks(range(1, len(data.labels) + 1))
            ax.set_xticklabels(data.labels)

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        if data.grid:
            ax.grid(True, alpha=0.3, axis='y' if data.vert else 'x')

        return fig

    def create_histogram(self, data: HistogramData) -> plt.Figure:
        """Crea un histograma usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(10, 6))

        n, bins, patches = ax.hist(
            data.data,
            bins=data.bins,
            density=data.density,
            cumulative=data.cumulative,
            alpha=data.alpha,
            color=data.color,
            edgecolor='black',
        )

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        if data.grid:
            ax.grid(True, alpha=0.3)

        return fig

    def create_pie_plot(self, data: PiePlotData) -> plt.Figure:
        """Crea un gráfico de pastel (pie chart) usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(10, 8))

        wedges, texts, autotexts = ax.pie(
            data.values,
            labels=data.labels,
            explode=data.explode,
            autopct=data.autopct,
            startangle=data.startangle,
            colors=data.colors,
            shadow=data.shadow,
        )

        ax.set_title(data.title)

        # Mejorar legibilidad de los textos
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        return fig

    def create_contour_plot(self, data: ContourPlotData) -> plt.Figure:
        """Crea un gráfico de contorno usando modelo de datos."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if data.filled:
            contour = ax.contourf(
                data.X,
                data.Y,
                data.Z,
                levels=data.levels,
                cmap=data.color_scheme.value,
            )
        else:
            contour = ax.contour(
                data.X,
                data.Y,
                data.Z,
                levels=data.levels,
                cmap=data.color_scheme.value,
            )

        if data.colorbar:
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label(data.colorbar_label)

        ax.set_xlabel(data.xlabel)
        ax.set_ylabel(data.ylabel)
        ax.set_title(data.title)

        return fig
