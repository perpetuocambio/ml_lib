# Módulo de Visualización - ml_lib.visualization

El módulo de visualización proporciona componentes generales de visualización basados en interfaces, servicios y handlers que pueden ser reutilizados en diferentes dominios, siguiendo los principios de arquitectura de ml_lib.

## Características

- **Interfaces estándar**: `VisualizationInterface`, `PlotTypeInterface`
- **Patrón de servicios**: `VisualizationService`, `PlottingService`
- **Gestión de errores**: `VisualizationErrorHandler`
- **Configuración centralizada**: `PlotConfig`
- **Tipado estricto**: Implementado con type hints de Python

## Estructura del módulo

```
visualization/
├── interfaces.py      # Interfaces estándar para componentes de visualización
├── models.py          # Modelos de datos para visualización (configuración, metadatos)
├── services.py        # Servicios para operaciones de visualización
├── handlers.py        # Handlers para manejo de errores, configuración y exportación
├── visualization.py   # Implementación concreta y fábrica de componentes
└── __init__.py        # Exportación de componentes públicos
```

## Componentes

### Interfaces
- `VisualizationInterface`: Interfaz base para componentes de visualización
- `PlotTypeInterface`: Interfaz para tipos específicos de gráficos

### Modelos
- `PlotConfig`: Configuración para componentes de visualización
- `VisualizationMetadata`: Metadatos para componentes de visualización

### Servicios
- `VisualizationService`: Servicio para operaciones generales de visualización
- `PlottingService`: Servicio para crear tipos específicos de gráficos

### Handlers
- `VisualizationErrorHandler`: Manejo de errores en operaciones de visualización
- `VisualizationConfigHandler`: Manejo de configuración de visualización
- `ImageExportHandler`: Manejo de exportación de imágenes

### Implementaciones
- `GeneralVisualization`: Implementación concreta de componentes de visualización generales
- `VisualizationFactory`: Fábrica para crear componentes de visualización

## Ejemplo de uso

```python
from ml_lib.visualization import VisualizationFactory, PlotConfig

# Crear instancia de visualización con configuración específica
viz = VisualizationFactory.create_visualization(
    PlotConfig(
        title="Gráfico de Ejemplo",
        xlabel="X",
        ylabel="Y",
        style="seaborn"
    )
)

# Crear diferentes tipos de gráficos
scatter_fig = viz.plot_scatter(x_data, y_data, title="Dispersión")
line_fig = viz.plot_line(x_data, y_data, title="Líneas")
bar_fig = viz.plot_bar(categories, values, title="Barras", labels=labels)
heatmap_fig = viz.plot_heatmap(data, title="Heatmap")

# Guardar figuras
viz.save_plot(scatter_fig, "scatter.png")
```

## Integración con aplicaciones de dominio

El módulo está diseñado para ser utilizado por aplicaciones de dominio específico como EcoML Analyzer. Estas aplicaciones pueden usar los componentes generales y adaptarlos a sus necesidades específicas sin implementar lógica de visualización específica en el dominio.

## Principios de diseño

- **Agnóstico al dominio**: Componentes generales que pueden aplicarse a cualquier dominio
- **Extensible**: Fácil de extender con nuevos tipos de gráficos
- **Configurable**: Amplia capacidad de configuración a través de `PlotConfig`
- **Manejo de errores robusto**: Uso de handlers para manejar errores específicos
- **Tipado estricto**: Seguridad en tiempo de desarrollo
