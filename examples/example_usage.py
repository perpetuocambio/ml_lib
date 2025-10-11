"""
Ejemplo de uso de la biblioteca de ML - Demostración de integración
"""

import numpy as np
from typing import Any, Dict

# Importar componentes de nuestra biblioteca
from ml_lib.core import (
    EstimatorInterface,
    BaseModel,
    ModelConfig,
    ValidationService,
    ErrorHandler,
    LoggingService,
)
from ml_lib.linalg.services.linalg import (
    Matrix,
    BLASService,
    DecompositionService,
)


class LinearRegressionEstimator(EstimatorInterface[np.ndarray, np.ndarray]):
    """
    Estimador de regresión lineal que demuestra el uso de los componentes de la biblioteca.
    """

    def __init__(self, config: ModelConfig = None):
        # Configuración del modelo
        self.config = config or ModelConfig()

        # Inicializar componentes de la biblioteca
        self.logger_service = LoggingService("LinearRegressionEstimator")
        self.logger = self.logger_service.get_logger()
        self.error_handler = ErrorHandler(self.logger)
        self.validation_service = ValidationService(self.logger)

        # Parámetros del modelo
        self.weights: np.ndarray = None
        self.bias: float = 0.0

        # Modelo base
        self.model_metadata = BaseModel(
            name="LinearRegression",
            version="1.0.0",
            metadata={"algorithm": "Ordinary Least Squares"},
        )

    @property
    def _fit_impl(self):
        return self.error_handler.handle_execution_error(self._fit_impl_raw)

    def _fit_impl_raw(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> "LinearRegressionEstimator":
        """Implementación interna del ajuste."""
        # Validar entradas
        self.validation_service.validate_input_shape(X, 2, "LinearRegression.fit")
        self.validation_service.validate_input_shape(y, 1, "LinearRegression.fit")
        self.validation_service.validate_same_length(X, y, "LinearRegression.fit")
        self.validation_service.validate_not_nan_inf(X, "LinearRegression.fit - X")
        self.validation_service.validate_not_nan_inf(y, "LinearRegression.fit - y")

        # Convertir a objetos Matrix para usar operaciones optimizadas
        X_matrix = Matrix(X)
        y_matrix = Matrix(y.reshape(-1, 1))  # Convertir a columna

        # Validar matrices
        X_matrix.validate_matrix()
        y_matrix.validate_matrix()

        # Añadir columna de unos para el sesgo (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Resolver usando descomposición SVD para estabilidad numérica
        # Este es un ejemplo de cómo se usaría el módulo linalg
        X_matrix_extended = Matrix(X_with_bias)
        XTX = BLASService.gemm(X_with_bias.T, X_with_bias)  # Producto X^T * X

        # Resolver el sistema lineal: (X^T * X) * theta = X^T * y
        XTy = BLASService.gemm(X_with_bias.T, y.reshape(-1, 1))

        # Usar descomposición para resolver
        U, S, Vh = DecompositionService.svd_decomposition(XTX)

        # Calcular pseudoinversa
        S_inv = np.diag([1 / s if s > 1e-10 else 0 for s in S.diagonal()])
        XTX_inv = Vh.T @ S_inv @ U.T

        # Calcular coeficientes
        theta = XTX_inv @ XTy

        # Separar pesos y sesgo
        self.bias = theta[0, 0]
        self.weights = theta[1:, 0]

        # Marcar el modelo como ajustado
        self.model_metadata.mark_fitted()

        self.logger.info(
            f"Modelo ajustado con éxito: weights shape = {self.weights.shape}"
        )

        return self

    def fit(
        self, X: np.ndarray, y: np.ndarray, **kwargs
    ) -> "LinearRegressionEstimator":
        """Ajusta el modelo a los datos."""
        return self._fit_impl(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones usando el modelo ajustado."""
        if not self.model_metadata.is_fitted:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones")

        # Validar entradas
        self.validation_service.validate_input_shape(X, 2, "LinearRegression.predict")

        # Validar que las dimensiones coincidan con el entrenamiento
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"La dimensionalidad de entrada {X.shape[1]} no coincide con la dimensionalidad del modelo {self.weights.shape[0]}"
            )

        # Realizar predicción
        predictions = X @ self.weights + self.bias

        self.logger.info(f"Realizadas {len(predictions)} predicciones")
        return predictions

    def get_params(self) -> Dict[str, Any]:
        """Obtiene los parámetros del modelo."""
        return {"weights": self.weights, "bias": self.bias, "config": self.config}

    def set_params(self, **params) -> "LinearRegressionEstimator":
        """Establece los parámetros del modelo."""
        if "weights" in params:
            self.weights = params["weights"]
        if "bias" in params:
            self.bias = params["bias"]
        if "config" in params:
            self.config = params["config"]
        return self


def main():
    """Función principal para demostrar el uso de la biblioteca."""
    print("Demostración de la biblioteca ML")
    print("=" * 40)

    # Generar datos de ejemplo
    np.random.seed(42)
    X = np.random.randn(100, 3)  # 100 muestras, 3 características
    true_weights = np.array([1.5, -2.0, 0.5])
    true_bias = 1.0
    y = (
        X @ true_weights + true_bias + 0.1 * np.random.randn(100)
    )  # Añadir algo de ruido

    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")

    # Crear y entrenar el modelo
    model = LinearRegressionEstimator()

    print("\nEntrenando el modelo...")
    model.fit(X, y)

    print(f"Pesos verdaderos: {true_weights}")
    print(f"Pesos estimados: {model.weights}")
    print(f"Sesgo verdadero: {true_bias:.3f}")
    print(f"Sesgo estimado: {model.bias:.3f}")

    # Realizar predicciones
    print("\nRealizando predicciones...")
    predictions = model.predict(X)

    # Calcular error
    mse = np.mean((y - predictions) ** 2)
    print(f"Error cuadrático medio en entrenamiento: {mse:.6f}")

    print("\n¡Demostración completada exitosamente!")


if __name__ == "__main__":
    main()
