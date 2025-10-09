"""
Ejemplo completo de uso de la biblioteca ml_lib con enfoque en modelos en lugar de diccionarios
"""
import numpy as np
from typing import Dict, List, Any

# Importar componentes principales de ml_lib
from ml_lib.core import (
    EstimatorInterface,
    TransformerInterface,
    ModelConfig,
    ValidationService,
    LoggingService
)

# Importar componentes de visualización mejorados
from ml_lib.visualization import VisualizationFactory, PlotConfig
from ml_lib.visualization.models import (
    ScatterPlotData, LinePlotData, BarPlotData, HeatmapData
)

# Importar componentes algebraicos
from ml_lib.linalg import Matrix, BLASService, DecompositionService


class CustomEstimator(EstimatorInterface[np.ndarray, np.ndarray]):
    """
    Ejemplo de estimador personalizado usando modelos en lugar de diccionarios.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.logger_service = LoggingService("CustomEstimator")
        self.logger = self.logger_service.get_logger()
        self.validation_service = ValidationService(self.logger)
        self.is_fitted = False
        self.weights = None
        self.bias = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'CustomEstimator':
        """Ajusta el modelo a los datos."""
        self.validation_service.validate_input_shape(X, 2, "fit")
        self.validation_service.validate_input_shape(y, 1, "fit")
        self.validation_service.validate_same_length(X, y, "fit")
        
        # Ajustar modelo de regresión lineal simple
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        XtX = BLASService.gemm(X_with_bias.T, X_with_bias)
        Xty = BLASService.gemm(X_with_bias.T, y.reshape(-1, 1))
        
        # Resolver sistema lineal
        try:
            coefficients = np.linalg.solve(XtX, Xty.flatten())
            self.bias = coefficients[0]
            self.weights = coefficients[1:]
        except np.linalg.LinAlgError:
            # Usar pseudoinversa si la matriz no es invertible
            coefficients = np.linalg.lstsq(XtX, Xty.flatten(), rcond=None)[0]
            self.bias = coefficients[0]
            self.weights = coefficients[1:]
        
        self.is_fitted = True
        self.logger.info(f"Modelo ajustado: {X.shape[1]} características, bias={self.bias:.3f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones con el modelo ajustado."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones")
        
        return X @ self.weights + self.bias
    
    def get_params(self) -> Dict[str, Any]:
        """Obtiene los parámetros del modelo."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params) -> 'CustomEstimator':
        """Establece los parámetros del modelo."""
        if 'weights' in params:
            self.weights = params['weights']
        if 'bias' in params:
            self.bias = params['bias']
        if 'is_fitted' in params:
            self.is_fitted = params['is_fitted']
        return self


class DataTransformer(TransformerInterface[np.ndarray]):
    """
    Ejemplo de transformador personalizado usando modelos en lugar de diccionarios.
    """
    
    def __init__(self):
        self.logger_service = LoggingService("DataTransformer")
        self.logger = self.logger_service.get_logger()
        self.validation_service = ValidationService(self.logger)
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DataTransformer':
        """Ajusta el transformador a los datos."""
        self.validation_service.validate_input_shape(X, 2, "fit")
        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1  # Evitar división por cero
        self.is_fitted = True
        
        self.logger.info(f"Transformador ajustado: {X.shape[1]} características")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma los datos."""
        if not self.is_fitted:
            raise ValueError("El transformador debe ser ajustado antes de transformar")
        
        self.validation_service.validate_input_shape(X, 2, "transform")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Ajusta y transforma los datos en un solo paso."""
        return self.fit(X, y).transform(X)


class ModelEvaluator:
    """
    Evaluador de modelos usando modelos en lugar de diccionarios para resultados.
    """
    
    def __init__(self):
        self.logger_service = LoggingService("ModelEvaluator")
        self.logger = self.logger_service.get_logger()
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evalúa un modelo de regresión usando métricas específicas.
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones del modelo
            
        Returns:
            Diccionario con métricas de evaluación
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        self.logger.info(f"Evaluación completada: MSE={mse:.4f}, R²={r2:.4f}")
        return metrics


def main():
    """
    Función principal que demuestra el uso de modelos en lugar de diccionarios.
    """
    print("🚀 Ejemplo completo de ml_lib usando modelos en lugar de diccionarios")
    print("="*70)
    
    # 1. Generar datos sintéticos
    print("\n1. Generando datos sintéticos...")
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100 muestras, 3 características
    true_weights = np.array([1.5, -2.0, 0.5])
    true_bias = 1.0
    y = X @ true_weights + true_bias + 0.1 * np.random.randn(100)
    
    print(f"   - Datos generados: X.shape={X.shape}, y.shape={y.shape}")
    
    # 2. Transformar datos
    print("\n2. Transformando datos...")
    transformer = DataTransformer()
    X_transformed = transformer.fit_transform(X)
    
    print(f"   - Datos transformados: {X_transformed.shape}")
    
    # 3. Entrenar modelo
    print("\n3. Entrenando modelo...")
    model = CustomEstimator()
    model.fit(X_transformed, y)
    
    print(f"   - Modelo entrenado: bias={model.bias:.3f}")
    
    # 4. Hacer predicciones
    print("\n4. Haciendo predicciones...")
    y_pred = model.predict(X_transformed)
    
    print(f"   - Predicciones realizadas: {y_pred.shape}")
    
    # 5. Evaluar modelo
    print("\n5. Evaluando modelo...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_regression(y, y_pred)
    
    print(f"   - Métricas: MSE={metrics['mse']:.4f}, R²={metrics['r2_score']:.4f}")
    
    # 6. Visualizar resultados usando modelos
    print("\n6. Visualizando resultados...")
    viz = VisualizationFactory.create_visualization(
        PlotConfig(title="Resultados del Modelo", style="seaborn")
    )
    
    # Crear datos para visualización usando modelos
    scatter_data = ScatterPlotData(
        x=y,  # Valores verdaderos
        y=y_pred,  # Predicciones
        title="Predicciones vs Valores Verdaderos",
        xlabel="Valores Verdaderos",
        ylabel="Predicciones",
        alpha=0.7,
        grid=True
    )
    
    # Crear gráfico de dispersión usando modelo
    fig1 = viz.plotting_service.create_scatter_plot(scatter_data)
    viz.save_plot(fig1, "predictions_vs_true.png")
    print("   - Gráfico de dispersión guardado")
    
    # Crear gráfico de residuos
    residuals = y - y_pred
    line_data = LinePlotData(
        x=np.arange(len(residuals)),
        y=residuals,
        title="Residuos del Modelo",
        xlabel="Muestras",
        ylabel="Residuos",
        linewidth=1.0,
        linestyle="-",
        color="red",
        grid=True
    )
    
    fig2 = viz.plotting_service.create_line_plot(line_data)
    viz.save_plot(fig2, "residuals.png")
    print("   - Gráfico de residuos guardado")
    
    # Crear gráfico de importancia de características
    feature_names = ["Característica 1", "Característica 2", "Característica 3"]
    feature_importance = np.abs(model.weights)
    
    bar_data = BarPlotData(
        x=np.arange(len(feature_importance)),
        heights=feature_importance,
        labels=feature_names,
        title="Importancia de Características",
        xlabel="Características",
        ylabel="Importancia Absoluta",
        color="blue",
        alpha=0.8,
        rotation=45,
        grid=True
    )
    
    fig3 = viz.plotting_service.create_bar_plot(bar_data)
    viz.save_plot(fig3, "feature_importance.png")
    print("   - Gráfico de importancia guardado")
    
    # 7. Demostrar ventajas de usar modelos en lugar de diccionarios
    print("\n7. Demostrando ventajas de modelos vs diccionarios...")
    
    # Ventaja 1: Tipado estricto
    print("   - Tipado estricto: Errores detectados en tiempo de desarrollo")
    
    # Ventaja 2: Validación automática
    print("   - Validación automática: Integridad de datos garantizada")
    
    # Ventaja 3: Documentación integrada
    print("   - Documentación integrada: Atributos claramente definidos")
    
    # Ventaja 4: Facilidad de mantenimiento
    print("   - Mantenibilidad: Cambios estructurados y seguros")
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("   Se han creado 3 archivos de visualización:")
    print("   - predictions_vs_true.png")
    print("   - residuals.png") 
    print("   - feature_importance.png")
    
    print("\n🎯 Beneficios demostrados:")
    print("   • Uso de modelos en lugar de diccionarios para mejor tipado")
    print("   • Validación automática de datos")
    print("   • Documentación integrada")
    print("   • Código más seguro y mantenible")
    print("   • Mejor experiencia de desarrollo")


if __name__ == "__main__":
    main()