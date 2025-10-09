"""
Handlers para álgebra lineal en ml_lib
"""

from typing import Callable, TypeVar, ParamSpec
from functools import wraps
import logging
import numpy as np
from scipy.linalg import LinAlgError

# Importar modelos
from .models import MatrixOperationConfig


P = ParamSpec("P")
R = TypeVar("R")


class LinearAlgebraErrorHandler:
    """Handler para manejo de errores en álgebra lineal."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def handle_linalg_error(self, func: Callable[P, R]) -> Callable[P, R]:
        """Decorador para manejo de errores en operaciones de álgebra lineal."""

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError) as e:
                self.logger.error(f"Error de validación en {func.__name__}: {str(e)}")
                raise
            except (LinAlgError, np.linalg.LinAlgError) as e:
                self.logger.error(
                    f"Error de álgebra lineal en {func.__name__}: {str(e)}"
                )
                raise
            except MemoryError as e:
                self.logger.critical(f"Error de memoria en {func.__name__}: {str(e)}")
                raise
            except Exception as e:
                self.logger.error(
                    f"Error inesperado en {func.__name__}: {str(e)}", exc_info=True
                )
                raise

        return wrapper


class MatrixConfigHandler:
    """Handler para manejo de configuración de matrices."""

    @staticmethod
    def validate_matrix_shape(matrix: np.ndarray, expected_dims: int = None) -> bool:
        """Valida la forma de una matriz."""
        if expected_dims and matrix.ndim != expected_dims:
            raise ValueError(
                f"Se esperaban {expected_dims} dimensiones, se recibieron {matrix.ndim}"
            )
        return True

    @staticmethod
    def sanitize_matrix_config(config: MatrixOperationConfig) -> MatrixOperationConfig:
        """Sanitiza la configuración de operaciones matriciales."""
        # Asegurar valores razonables
        config.num_threads = max(1, min(64, config.num_threads))  # Limitar hilos
        config.block_size = max(8, min(1024, config.block_size))  # Bloques razonables
        config.memory_alignment = max(
            8, min(256, config.memory_alignment)
        )  # Alineación

        # Validar conjunto de instrucciones SIMD
        valid_simd_sets = ["SSE", "AVX", "AVX2", "AVX512"]
        if config.simd_instruction_set not in valid_simd_sets:
            config.simd_instruction_set = "AVX2"  # Valor por defecto

        # Validar nivel de optimización
        valid_opt_levels = ["O0", "O1", "O2", "O3"]
        if config.optimization_level not in valid_opt_levels:
            config.optimization_level = "O2"

        return config


class MemoryLayoutHandler:
    """Handler para manejo de layouts de memoria."""

    @staticmethod
    def ensure_c_contiguous(matrix: np.ndarray) -> np.ndarray:
        """Asegura que la matriz sea C-contigua."""
        if not matrix.flags.c_contiguous:
            return np.ascontiguousarray(matrix)
        return matrix

    @staticmethod
    def ensure_f_contiguous(matrix: np.ndarray) -> np.ndarray:
        """Asegura que la matriz sea F-contigua."""
        if not matrix.flags.f_contiguous:
            return np.asfortranarray(matrix)
        return matrix

    @staticmethod
    def get_optimal_layout(matrix: np.ndarray, operation: str = "general") -> str:
        """Determina el layout óptimo para una operación."""
        # Para operaciones generales, C-contiguous suele ser óptimo
        if operation == "general":
            return "C"
        # Para operaciones específicas como BLAS, puede depender
        elif operation == "blas":
            return "F" if matrix.flags.f_contiguous else "C"
        else:
            return "C"


class PrecisionHandler:
    """Handler para manejo de precisión numérica."""

    @staticmethod
    def get_precision_dtype(precision: str = "double") -> np.dtype:
        """Obtiene el tipo de dato apropiado para una precisión."""
        if precision == "single":
            return np.float32
        elif precision == "double":
            return np.float64
        elif precision == "extended":
            return np.longdouble
        else:
            return np.float64

    @staticmethod
    def cast_to_precision(matrix: np.ndarray, precision: str = "double") -> np.ndarray:
        """Convierte una matriz a una precisión específica."""
        target_dtype = PrecisionHandler.get_precision_dtype(precision)
        if matrix.dtype != target_dtype:
            return matrix.astype(target_dtype)
        return matrix


class MatrixValidationHandler:
    """Handler para validación de matrices."""

    @staticmethod
    def validate_square_matrix(matrix: np.ndarray) -> None:
        """Valida que una matriz sea cuadrada."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"La matriz debe ser cuadrada, pero es de forma {matrix.shape}"
            )

    @staticmethod
    def validate_positive_definite(matrix: np.ndarray) -> None:
        """Valida que una matriz sea definida positiva."""
        MatrixValidationHandler.validate_square_matrix(matrix)

        # Verificar simetría
        if not np.allclose(matrix, matrix.T):
            raise ValueError("La matriz debe ser simétrica para ser definida positiva")

        # Verificar valores propios positivos
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            if np.any(eigenvalues <= 0):
                raise ValueError(
                    "La matriz no es definida positiva (tiene valores propios no positivos)"
                )
        except np.linalg.LinAlgError:
            raise ValueError(
                "No se pudieron calcular los valores propios para verificar definida positiva"
            )

    @staticmethod
    def validate_invertible(matrix: np.ndarray, rtol: float = 1e-10) -> None:
        """Valida que una matriz sea invertible."""
        try:
            det = np.linalg.det(matrix)
            if abs(det) < rtol:
                raise ValueError(
                    f"La matriz no es invertible (determinante ≈ {det:.2e})"
                )
        except np.linalg.LinAlgError:
            raise ValueError(
                "No se pudo calcular el determinante para verificar invertibilidad"
            )
