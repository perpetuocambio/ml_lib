"""
Galería de visualización - Showcase de temas y plots elegantes

Este script demuestra todas las capacidades del módulo de visualización:
- 11 temas predefinidos elegantes
- 25+ esquemas de color
- 10 tipos de plots diferentes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importar componentes de ml_lib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_lib.visualization.models import (
    ScatterPlotData,
    LinePlotData,
    BarPlotData,
    HeatmapData,
    BoxPlotData,
    ViolinPlotData,
    HistogramData,
    PiePlotData,
    ContourPlotData,
    ColorScheme,
    AVAILABLE_THEMES,
)
from ml_lib.visualization.services.services import PlottingService
from ml_lib.visualization.services.theme_manager import get_theme_manager, apply_theme
import logging


def create_sample_data():
    """Genera datos de ejemplo para las visualizaciones."""
    np.random.seed(42)

    # Datos básicos
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)

    # Datos para scatter
    scatter_x = np.random.randn(50)
    scatter_y = np.random.randn(50)

    # Datos para bar
    categories = ['A', 'B', 'C', 'D', 'E']
    bar_heights = np.random.randint(10, 100, 5)

    # Datos para heatmap
    heatmap_data = np.random.randn(10, 10)

    # Datos para box y violin
    box_data = [np.random.randn(100) for _ in range(4)]

    # Datos para histogram
    hist_data = np.random.randn(1000)

    # Datos para pie
    pie_values = np.array([30, 25, 20, 15, 10])
    pie_labels = ['Segment A', 'Segment B', 'Segment C', 'Segment D', 'Segment E']

    # Datos para contour
    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = np.sin(np.sqrt(X**2 + Y**2))

    return {
        'line': (x, y),
        'scatter': (scatter_x, scatter_y),
        'bar': (categories, bar_heights),
        'heatmap': heatmap_data,
        'box': box_data,
        'violin': box_data,
        'histogram': hist_data,
        'pie': (pie_values, pie_labels),
        'contour': (X, Y, Z),
    }


def showcase_themes():
    """Muestra todos los temas disponibles."""
    print("=" * 80)
    print("GALERÍA DE TEMAS DE VISUALIZACIÓN")
    print("=" * 80)
    print()

    theme_manager = get_theme_manager()
    logger = logging.getLogger(__name__)
    plotting_service = PlottingService(logger)

    data = create_sample_data()
    x, y = data['line']

    output_dir = Path(__file__).parent / "output" / "themes"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Generando visualizaciones en: {output_dir}")
    print()

    for theme_name in AVAILABLE_THEMES.keys():
        print(f"🎨 Generando tema: {theme_name}")

        # Aplicar tema
        apply_theme(theme_name)

        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"Theme: {theme_name.replace('_', ' ').title()}",
                     fontsize=16, fontweight='bold')

        # 1. Line plot
        ax1 = plt.subplot(2, 3, 1)
        line_data = LinePlotData(
            x=x, y=y,
            title="Line Plot",
            xlabel="X axis", ylabel="Y axis"
        )
        # Renderizar manualmente en el subplot
        ax1.plot(line_data.x, line_data.y, linewidth=line_data.linewidth)
        ax1.set_title(line_data.title)
        ax1.set_xlabel(line_data.xlabel)
        ax1.set_ylabel(line_data.ylabel)
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plot
        ax2 = plt.subplot(2, 3, 2)
        scatter_x, scatter_y = data['scatter']
        ax2.scatter(scatter_x, scatter_y, alpha=0.6, s=50)
        ax2.set_title("Scatter Plot")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, alpha=0.3)

        # 3. Bar plot
        ax3 = plt.subplot(2, 3, 3)
        categories, bar_heights = data['bar']
        ax3.bar(range(len(bar_heights)), bar_heights, alpha=0.7)
        ax3.set_xticks(range(len(categories)))
        ax3.set_xticklabels(categories)
        ax3.set_title("Bar Plot")
        ax3.set_ylabel("Values")
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Histogram
        ax4 = plt.subplot(2, 3, 4)
        hist_data = data['histogram']
        ax4.hist(hist_data, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_title("Histogram")
        ax4.set_xlabel("Values")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)

        # 5. Box plot
        ax5 = plt.subplot(2, 3, 5)
        box_data = data['box']
        ax5.boxplot(box_data, patch_artist=True)
        ax5.set_title("Box Plot")
        ax5.set_xlabel("Categories")
        ax5.set_ylabel("Values")
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Heatmap
        ax6 = plt.subplot(2, 3, 6)
        heatmap_data = data['heatmap']
        im = ax6.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax6.set_title("Heatmap")
        plt.colorbar(im, ax=ax6, label="Value")

        plt.tight_layout()

        # Guardar
        output_file = output_dir / f"theme_{theme_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Guardado: {output_file.name}")

    print()
    print(f"✨ Galería de temas completada: {len(AVAILABLE_THEMES)} temas generados")
    print()


def showcase_color_schemes():
    """Muestra todos los esquemas de color disponibles."""
    print("=" * 80)
    print("GALERÍA DE ESQUEMAS DE COLOR")
    print("=" * 80)
    print()

    output_dir = Path(__file__).parent / "output" / "color_schemes"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Generando visualizaciones en: {output_dir}")
    print()

    # Datos para mostrar color schemes
    data = np.random.randn(10, 10)

    # Categorías de color schemes
    categories = {
        'Sequential': [
            ColorScheme.VIRIDIS, ColorScheme.PLASMA, ColorScheme.INFERNO,
            ColorScheme.MAGMA, ColorScheme.CIVIDIS
        ],
        'Diverging': [
            ColorScheme.COOLWARM, ColorScheme.RD_BU, ColorScheme.RD_GY,
            ColorScheme.BR_BG, ColorScheme.PI_YG, ColorScheme.SPECTRAL
        ],
        'Qualitative': [
            ColorScheme.TAB10, ColorScheme.TAB20, ColorScheme.SET1,
            ColorScheme.SET2, ColorScheme.SET3, ColorScheme.PAIRED
        ],
        'Perceptually Uniform': [
            ColorScheme.ROCKET, ColorScheme.MAKO, ColorScheme.FLARE,
            ColorScheme.CREST
        ],
        'Other': [
            ColorScheme.RD_YL_GN, ColorScheme.RD_YL_BU
        ]
    }

    for category, schemes in categories.items():
        print(f"🎨 Categoría: {category}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Color Schemes - {category}", fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for idx, scheme in enumerate(schemes):
            if idx >= len(axes):
                break

            ax = axes[idx]
            im = ax.imshow(data, cmap=scheme.value, aspect='auto')
            ax.set_title(scheme.name)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Ocultar ejes sobrantes
        for idx in range(len(schemes), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Guardar
        output_file = output_dir / f"colorscheme_{category.lower().replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✅ Guardado: {output_file.name}")

    print()
    print(f"✨ Galería de color schemes completada")
    print()


def showcase_plot_types():
    """Muestra todos los tipos de plots disponibles con el tema Material."""
    print("=" * 80)
    print("GALERÍA DE TIPOS DE PLOTS")
    print("=" * 80)
    print()

    # Aplicar tema Material
    apply_theme('material')

    logger = logging.getLogger(__name__)
    plotting_service = PlottingService(logger)

    data = create_sample_data()

    output_dir = Path(__file__).parent / "output" / "plot_types"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Generando visualizaciones en: {output_dir}")
    print()

    # 1. Line Plot
    print("📊 Generando: Line Plot")
    x, y = data['line']
    line_data = LinePlotData(x=x, y=y, title="Line Plot Example")
    fig = plotting_service.create_line_plot(line_data)
    fig.savefig(output_dir / "plot_line.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Scatter Plot
    print("📊 Generando: Scatter Plot")
    scatter_x, scatter_y = data['scatter']
    scatter_data = ScatterPlotData(x=scatter_x, y=scatter_y, title="Scatter Plot Example")
    fig = plotting_service.create_scatter_plot(scatter_data)
    fig.savefig(output_dir / "plot_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Bar Plot
    print("📊 Generando: Bar Plot")
    categories, bar_heights = data['bar']
    bar_data = BarPlotData(
        x=np.arange(len(bar_heights)),
        heights=bar_heights,
        labels=categories,
        title="Bar Plot Example"
    )
    fig = plotting_service.create_bar_plot(bar_data)
    fig.savefig(output_dir / "plot_bar.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Heatmap
    print("📊 Generando: Heatmap")
    heatmap_matrix = data['heatmap']
    heatmap_data = HeatmapData(data=heatmap_matrix, title="Heatmap Example")
    fig = plotting_service.create_heatmap(heatmap_data)
    fig.savefig(output_dir / "plot_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Box Plot
    print("📊 Generando: Box Plot")
    box_list = data['box']
    box_data = BoxPlotData(
        data=box_list,
        labels=['Group A', 'Group B', 'Group C', 'Group D'],
        title="Box Plot Example"
    )
    fig = plotting_service.create_box_plot(box_data)
    fig.savefig(output_dir / "plot_box.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Violin Plot
    print("📊 Generando: Violin Plot")
    violin_list = data['violin']
    violin_data = ViolinPlotData(
        data=violin_list,
        labels=['Group A', 'Group B', 'Group C', 'Group D'],
        title="Violin Plot Example"
    )
    fig = plotting_service.create_violin_plot(violin_data)
    fig.savefig(output_dir / "plot_violin.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Histogram
    print("📊 Generando: Histogram")
    hist_array = data['histogram']
    histogram_data = HistogramData(data=hist_array, bins=50, title="Histogram Example")
    fig = plotting_service.create_histogram(histogram_data)
    fig.savefig(output_dir / "plot_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 8. Pie Plot
    print("📊 Generando: Pie Plot")
    pie_values, pie_labels = data['pie']
    pie_data = PiePlotData(
        values=pie_values,
        labels=pie_labels,
        title="Pie Chart Example"
    )
    fig = plotting_service.create_pie_plot(pie_data)
    fig.savefig(output_dir / "plot_pie.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 9. Contour Plot
    print("📊 Generando: Contour Plot")
    X, Y, Z = data['contour']
    contour_data = ContourPlotData(
        X=X, Y=Y, Z=Z,
        title="Contour Plot Example",
        filled=True
    )
    fig = plotting_service.create_contour_plot(contour_data)
    fig.savefig(output_dir / "plot_contour.png", dpi=150, bbox_inches='tight')
    plt.close()

    print()
    print(f"✨ Galería de plots completada: 9 tipos generados")
    print()


def show_theme_preview():
    """Muestra un preview de colores de cada tema."""
    print("=" * 80)
    print("PREVIEW DE PALETAS DE COLOR")
    print("=" * 80)
    print()

    theme_manager = get_theme_manager()

    for theme_name in AVAILABLE_THEMES.keys():
        palette = theme_manager.get_theme_preview(theme_name)

        print(f"🎨 {theme_name.replace('_', ' ').title()}")
        print(f"   Primary:    {palette['primary']}")
        print(f"   Secondary:  {palette['secondary']}")
        print(f"   Accent:     {palette['accent']}")
        print(f"   Background: {palette['background']}")
        print()


def main():
    """Ejecuta todas las galerías."""
    logging.basicConfig(level=logging.WARNING)

    print()
    print("🎨" * 40)
    print("   GALERÍA COMPLETA DE VISUALIZACIÓN ML_LIB")
    print("🎨" * 40)
    print()

    # Preview de paletas
    show_theme_preview()

    # Generar galerías
    showcase_themes()
    showcase_color_schemes()
    showcase_plot_types()

    print()
    print("=" * 80)
    print("✨ GALERÍA COMPLETA GENERADA EXITOSAMENTE ✨")
    print("=" * 80)
    print()
    print("📂 Revisa los archivos generados en:")
    print("   • examples/output/themes/")
    print("   • examples/output/color_schemes/")
    print("   • examples/output/plot_types/")
    print()


if __name__ == "__main__":
    main()
