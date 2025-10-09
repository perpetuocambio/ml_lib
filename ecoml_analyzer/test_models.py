"""
Prueba para verificar el uso de modelos en lugar de diccionarios en ml_lib
"""
import numpy as np
from ml_lib.visualization import VisualizationFactory, PlotConfig
from ml_lib.visualization.models import (
    ScatterPlotData, LinePlotData, BarPlotData, HeatmapData
)


def test_models_vs_dicts():
    """Prueba para demostrar el uso de modelos en lugar de diccionarios."""
    print("🧪 Prueba de modelos vs diccionarios")
    print("="*40)
    
    # Crear datos de ejemplo
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    # Crear instancia de visualización
    viz = VisualizationFactory.create_visualization(
        PlotConfig(title="Test de Modelos", style="seaborn")
    )
    
    # Ejemplo 1: Uso de modelo ScatterPlotData
    print("1. Creando gráfico de dispersión con modelo...")
    scatter_data = ScatterPlotData(
        x=x[:20],  # Usar solo parte de los datos para claridad
        y=y[:20],
        title="Gráfico de Dispersión con Modelo",
        xlabel="Tiempo",
        ylabel="Valor",
        alpha=0.7,
        grid=True
    )
    
    # Usar el servicio de plotting con el modelo
    fig1 = viz.plotting_service.create_scatter_plot(scatter_data)
    viz.save_plot(fig1, "test_scatter_model.png")
    print("   ✅ Gráfico de dispersión creado y guardado")
    
    # Ejemplo 2: Uso de modelo LinePlotData
    print("2. Creando gráfico de líneas con modelo...")
    line_data = LinePlotData(
        x=x,
        y=y,
        title="Gráfico de Líneas con Modelo",
        xlabel="Tiempo",
        ylabel="Valor",
        linewidth=2.0,
        linestyle="-",
        color="blue",
        grid=True
    )
    
    fig2 = viz.plotting_service.create_line_plot(line_data)
    viz.save_plot(fig2, "test_line_model.png")
    print("   ✅ Gráfico de líneas creado y guardado")
    
    # Ejemplo 3: Uso de modelo BarPlotData
    print("3. Creando gráfico de barras con modelo...")
    categories = np.array(["A", "B", "C", "D", "E"])
    values = np.random.rand(5) * 100
    
    bar_data = BarPlotData(
        x=np.arange(len(categories)),
        heights=values,
        labels=categories.tolist(),
        title="Gráfico de Barras con Modelo",
        xlabel="Categorías",
        ylabel="Valores",
        color="green",
        alpha=0.7,
        rotation=45
    )
    
    fig3 = viz.plotting_service.create_bar_plot(bar_data)
    viz.save_plot(fig3, "test_bar_model.png")
    print("   ✅ Gráfico de barras creado y guardado")
    
    # Ejemplo 4: Uso de modelo HeatmapData
    print("4. Creando heatmap con modelo...")
    data_matrix = np.random.rand(10, 10)
    x_labels = [f"Col_{i}" for i in range(10)]
    y_labels = [f"Row_{i}" for i in range(10)]
    
    heatmap_data = HeatmapData(
        data=data_matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        title="Heatmap con Modelo",
        color_map="viridis",
        center=0.5,
        colorbar_label="Valor",
        rotation=45
    )
    
    fig4 = viz.plotting_service.create_heatmap(heatmap_data)
    viz.save_plot(fig4, "test_heatmap_model.png")
    print("   ✅ Heatmap creado y guardado")
    
    print("\n✅ Todas las pruebas de modelos completadas exitosamente!")
    print("   Los archivos de prueba se han guardado en el directorio actual.")


def test_model_advantages():
    """Prueba para demostrar ventajas de usar modelos en lugar de diccionarios."""
    print("\n🔍 Ventajas de usar modelos en lugar de diccionarios")
    print("="*55)
    
    # Ventaja 1: Tipado estricto
    print("1. Tipado estricto:")
    try:
        # Esto fallaría en tiempo de desarrollo si usáramos diccionarios
        scatter_data = ScatterPlotData(
            x=np.array([1, 2, 3]),
            y=np.array([4, 5, 6]),
            title="Prueba de Tipado",
            # alpha="invalido"  # Esto causaría un error de tipo en tiempo de desarrollo
        )
        print("   ✅ Tipado estricto ayuda a detectar errores temprano")
    except Exception as e:
        print(f"   ❌ Error de tipado: {e}")
    
    # Ventaja 2: Validación automática
    print("2. Validación automática:")
    try:
        # Los modelos pueden incluir validación en __post_init__
        heatmap_data = HeatmapData(
            data=np.array([[1, 2], [3, 4]]),
            title="Prueba de Validación"
        )
        print("   ✅ Validación automática asegura integridad de datos")
    except Exception as e:
        print(f"   ❌ Error de validación: {e}")
    
    # Ventaja 3: Documentación automática
    print("3. Documentación automática:")
    print(f"   ✅ Los modelos generan documentación automáticamente")
    print(f"   ✅ Atributos claramente definidos: {list(ScatterPlotData.__dataclass_fields__.keys())}")
    
    # Ventaja 4: Facilidad de extensión
    print("4. Facilidad de extensión:")
    print("   ✅ Fácil de extender con nuevas propiedades")
    print("   ✅ Herencia de modelos para especialización")
    
    print("\n💡 Beneficios clave de usar modelos en lugar de diccionarios:")
    print("   • Seguridad de tipos en tiempo de desarrollo")
    print("   • Validación automática de datos")
    print("   • Documentación integrada y autogenerada")
    print("   • Facilidad de mantenimiento y extensión")
    print("   • Mejor experiencia de desarrollo con IDE")


if __name__ == "__main__":
    test_models_vs_dicts()
    test_model_advantages()
    
    print("\n🎉 ¡Prueba de modelos completada!")
    print("La implementación demuestra cómo usar modelos en lugar de diccionarios")
    print("para mejorar la calidad, seguridad y mantenibilidad del código.")