"""
Tests básicos para verificar la funcionalidad de la biblioteca
"""
import numpy as np
import pytest
from ml_lib.core import (
    EstimatorInterface,
    BaseModel,
    ModelConfig,
    ValidationService,
    ErrorHandler,
    LoggingService
)
from ml_lib.linalg import (
    Matrix,
    BLASService,
    DecompositionService
)
from example_usage import LinearRegressionEstimator


def test_matrix_creation():
    """Test de creación de matrices."""
    data = np.array([[1, 2], [3, 4]])
    matrix = Matrix(data)
    
    assert matrix.shape == (2, 2)
    assert matrix.dtype == data.dtype
    assert np.array_equal(matrix.data, data)
    
    # Test de validación
    matrix.validate_matrix()  # No debería lanzar excepción


def test_matrix_with_nan():
    """Test de validación de matriz con NaN."""
    data = np.array([[1, 2], [np.nan, 4]])
    matrix = Matrix(data)
    
    with pytest.raises(ValueError, match="Matrix contains NaN values"):
        matrix.validate_matrix()


def test_blas_service():
    """Test de servicio BLAS."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[2.0, 0.0], [1.0, 2.0]])
    
    result = BLASService.gemm(A, B)
    expected = np.dot(A, B)
    
    assert np.allclose(result, expected)


def test_decomposition_service():
    """Test de servicio de descomposición."""
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    
    U, S, Vh = DecompositionService.svd_decomposition(A)
    
    # Verificar que la reconstrucción es correcta
    reconstructed = U @ S @ Vh
    assert np.allclose(reconstructed, A, atol=1e-10)


def test_linear_regression_estimator():
    """Test del estimador de regresión lineal."""
    # Generar datos de ejemplo
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(20)
    
    # Crear y entrenar modelo
    model = LinearRegressionEstimator()
    model.fit(X, y)
    
    # Verificar que el modelo fue ajustado
    assert model.weights is not None
    assert model.bias is not None
    assert model.weights.shape[0] == 2  # 2 características
    
    # Realizar predicciones
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    
    # Verificar que las predicciones tienen sentido
    mse = np.mean((y - predictions) ** 2)
    assert mse < 1.0  # Debería tener un error razonablemente bajo


def test_estimator_interface():
    """Test de la interfaz del estimador."""
    estimator = LinearRegressionEstimator()
    
    # Verificar que implementa la interfaz
    assert isinstance(estimator, EstimatorInterface)
    
    # Verificar que tiene los métodos necesarios
    assert hasattr(estimator, 'fit')
    assert hasattr(estimator, 'predict')
    assert hasattr(estimator, 'get_params')
    assert hasattr(estimator, 'set_params')


def test_model_validation():
    """Test del servicio de validación."""
    logger_service = LoggingService("test")
    logger = logger_service.get_logger()
    validation_service = ValidationService(logger)
    
    # Test de validación de forma
    X = np.random.randn(10, 5)
    validation_service.validate_input_shape(X, 2, "test")  # Debería pasar
    
    with pytest.raises(ValueError, match="Expected 3D array, got 2D"):
        validation_service.validate_input_shape(X, 3, "test")
    
    # Test de validación de longitud
    y = np.random.randn(10)
    validation_service.validate_same_length(X, y, "test")  # Debería pasar
    
    y_bad = np.random.randn(9)
    with pytest.raises(ValueError, match="X and y must have same length"):
        validation_service.validate_same_length(X, y_bad, "test")


if __name__ == "__main__":
    # Ejecutar los tests manualmente
    test_matrix_creation()
    print("✓ test_matrix_creation")
    
    test_matrix_with_nan()
    print("✓ test_matrix_with_nan")
    
    test_blas_service()
    print("✓ test_blas_service")
    
    test_decomposition_service()
    print("✓ test_decomposition_service")
    
    test_linear_regression_estimator()
    print("✓ test_linear_regression_estimator")
    
    test_estimator_interface()
    print("✓ test_estimator_interface")
    
    test_model_validation()
    print("✓ test_model_validation")
    
    print("\n¡Todos los tests pasaron exitosamente!")