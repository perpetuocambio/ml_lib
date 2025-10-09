# ML Library

Una biblioteca de Machine Learning de alto rendimiento y código agnóstico escrita en Python, con tipado estricto y arquitectura modular.

## Características

- **Tipado estricto**: Implementado con type hints de Python para una experiencia de desarrollo segura
- **Arquitectura modular**: Diseño en módulos independientes para reutilización y extensibilidad
- **Alto rendimiento**: Optimizado para cómputo eficiente con NumPy y otras bibliotecas de bajo nivel
- **Interfaz consistente**: Interfaces estándar para todos los componentes de ML
- **Código agnóstico**: Diseñado para trabajar con diferentes backends y frameworks

## Estructura del Proyecto

```
ml_lib/
├── core/              # Componentes fundamentales
├── linalg/            # Operaciones de álgebra lineal
├── visualization/     # Componentes de visualización generales
├── autograd/          # Diferenciación automática
├── optimization/      # Algoritmos de optimización
├── kernels/           # Métodos de kernel
├── probabilistic/     # Modelos probabilísticos
├── neural/            # Redes neuronales
├── ensemble/          # Métodos de ensemble
├── feature_engineering/ # Ingeniería de características
├── data_processing/   # Procesamiento de datos
├── uncertainty/       # Cuantificación de incertidumbre
├── time_series/       # Modelado de series temporales
├── reinforcement/     # Aprendizaje por refuerzo
├── interpretability/  # Interpretación de modelos
├── automl/            # Automatización de ML
├── fairness/          # Equidad y sesgo
├── deployment/        # Despliegue de modelos
├── testing/           # Pruebas y validación
├── plugin_system/     # Sistema de plugins
├── performance/       # Rendimiento y optimización
├── ecoml_analyzer/    # Aplicación de demostración ecológica
└── utils/             # Utilidades generales
```

## Aplicación de demostración: EcoML Analyzer

Incluimos una aplicación de demostración completa llamada **EcoML Analyzer** que ejemplifica el uso de nuestra biblioteca en un contexto ecológico real. La aplicación:

- Analiza datos de abundancia de especies
- Realiza análisis de diversidad, comunidades ecológicas y distribución de especies
- Incluye visualización de resultados ecológicos
- Demuestra la integración de todos los componentes de la biblioteca

La aplicación utiliza componentes generales de la biblioteca (como los de visualización) y los aplica al dominio ecológico, demostrando el enfoque agnóstico al dominio de nuestra biblioteca.

Para ejecutar la demostración:

```bash
cd /src/perpetuocambio/ml_lib
PYTHONPATH=. python ecoml_analyzer/main.py
```

## Módulo de visualización general

El módulo `visualization` proporciona componentes de visualización basados en interfaces, servicios y handlers que pueden ser reutilizados en diferentes dominios. Incluye:

- **Interfaces**: `VisualizationInterface`, `PlotTypeInterface`
- **Modelos**: `PlotConfig`, `VisualizationMetadata`
- **Servicios**: `VisualizationService`, `PlottingService`
- **Handlers**: `VisualizationErrorHandler`, `ImageExportHandler`

## Instalación

```bash
uv venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
uv pip install -e .
```

## Uso

```python
from ml_lib.core import EstimatorInterface
from ml_lib.visualization import VisualizationFactory, PlotConfig

# Ejemplo de uso de componentes generales
viz = VisualizationFactory.create_visualization(
    PlotConfig(title="Mi Gráfico", style="seaborn")
)

fig = viz.plot_scatter(x_data, y_data)
viz.save_plot(fig, "mi_grafico.png")

# Ejemplo de implementación de estimador
class MyEstimator(EstimatorInterface):
    def fit(self, X, y, **kwargs):
        # Implementación del ajuste
        pass
    
    def predict(self, X):
        # Implementación de la predicción
        pass
    
    def get_params(self):
        # Obtener hiperparámetros
        pass
    
    def set_params(self, **params):
        # Establecer hiperparámetros
        pass
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerencias de mejora.

## Licencia

MIT