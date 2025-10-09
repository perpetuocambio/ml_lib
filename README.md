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

Para ejecutar la demostración:

```bash
cd /src/perpetuocambio/ml_lib
PYTHONPATH=. python ecoml_analyzer/main.py
```

## Instalación

```bash
uv venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
uv pip install -e .
```

## Uso

```python
from ml_lib.core import EstimatorInterface

# Ejemplo de uso
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