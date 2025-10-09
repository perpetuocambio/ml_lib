"""
Ejemplo de uso de los componentes generales de visualización de ml_lib
"""
import numpy as np
from ml_lib.visualization import VisualizationFactory, PlotConfig


def demo_general_visualization():
    """Demostración de componentes generales de visualización."""
    print("🎨 Demostración de componentes generales de visualización")
    print("="*60)
    
    # Crear datos de ejemplo
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    # Crear instancia de visualización general
    viz = VisualizationFactory.create_visualization(
        PlotConfig(
            title="Visualización General de Ejemplo",
            xlabel="X",
            ylabel="Y",
            style="seaborn"
        )
    )
    
    # Crear diferentes tipos de gráficos
    print("1. Creando gráfico de dispersión...")
    scatter_fig = viz.plot_scatter(x, y, title="Gráfico de Dispersión")
    
    print("2. Creando gráfico de líneas...")
    line_fig = viz.plot_line(x, y, title="Gráfico de Líneas")
    
    print("3. Creando gráfico de barras...")
    # Para el gráfico de barras, necesitamos datos categóricos
    categories = np.arange(5)
    values = np.random.rand(5) * 100
    bar_fig = viz.plot_bar(categories, values, 
                          title="Gráfico de Barras", 
                          labels=[f'Cat {i}' for i in categories])
    
    # Crear heatmap
    print("4. Creando heatmap...")
    data = np.random.rand(10, 10)
    heatmap_fig = viz.plot_heatmap(data, title="Heatmap de Ejemplo")
    
    # Guardar figuras
    print("5. Guardando figuras...")
    viz.save_plot(scatter_fig, "general_viz_scatter.png")
    viz.save_plot(line_fig, "general_viz_line.png")
    viz.save_plot(bar_fig, "general_viz_bar.png")
    viz.save_plot(heatmap_fig, "general_viz_heatmap.png")
    
    print("✅ Demostración de componentes generales de visualización completada")
    print("   Las figuras han sido guardadas en el directorio actual")


def demo_ecological_visualization_integration():
    """Demostración de cómo ecoml_analyzer usa componentes generales."""
    print("\n🌿 Integración de visualización ecológica con componentes generales")
    print("="*70)
    
    from ecoml_analyzer.visualization import EcologicalVisualizer
    
    # Simular algunos resultados de análisis ecológico
    n_sites = 20
    species_names = [f"Species_{i}" for i in range(10)]
    site_names = [f"Site_{i}" for i in range(n_sites)]
    
    # Matriz de abundancia simulada
    abundance_matrix = np.random.poisson(3, size=(10, n_sites))
    
    # Resultados de análisis de composición
    composition_results = {
        'richness': np.random.randint(1, 10, n_sites),
        'diversity': np.random.uniform(1, 5, n_sites),
        'total_values': np.sum(abundance_matrix, axis=0),
        'abundance_distribution': {
            'mean': np.mean(abundance_matrix),
            'std': np.std(abundance_matrix),
            'min': np.min(abundance_matrix),
            'max': np.max(abundance_matrix)
        }
    }
    
    # Crear visualizador ecológico que usa componentes generales
    eco_viz = EcologicalVisualizer(style="seaborn")
    
    print("1. Creando heatmap de abundancia ecológica...")
    heatmap_fig = eco_viz.plot_species_abundance_heatmap(
        abundance_matrix, 
        species_names, 
        site_names,
        title="Heatmap de Abundancia Ecológica"
    )
    
    print("2. Creando análisis de composición...")
    composition_fig = eco_viz.plot_composition_analysis(
        composition_results,
        site_names,
        title="Análisis de Composición Ecológica"
    )
    
    print("3. Guardando visualizaciones ecológicas...")
    eco_viz.save_all_figures("eco_viz_output")
    
    print("✅ Demostración de integración ecológica con componentes generales completada")
    print("   Las figuras han sido guardadas en el directorio eco_viz_output/")


if __name__ == "__main__":
    demo_general_visualization()
    demo_ecological_visualization_integration()
    
    print("\n🎉 ¡Demostración completada!")
    print("El ejemplo muestra cómo componentes generales de visualización") 
    print("se pueden aplicar a diferentes dominios, incluyendo ecología.")