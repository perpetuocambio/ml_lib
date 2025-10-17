"""
Modelo base con metadatos comunes y tipado estricto.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

from .enums import ModelState
from .metadata import ModelMetadata, PerformanceMetrics


@dataclass
class BaseModel:
    """Modelo base con metadatos comunes y estado fuertemente tipado."""

    name: str
    version: str
    state: ModelState = ModelState.INITIALIZED
    metadata: Optional[ModelMetadata] = None
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    created_at: str = field(init=False)
    updated_at: str = field(init=False)

    def __post_init__(self) -> None:
        """Inicialización post-creación."""
        now = datetime.now().isoformat()
        self.created_at = now
        self.updated_at = now

        # Crear metadata por defecto si no se proporciona
        if self.metadata is None:
            self.metadata = ModelMetadata(
                model_id=f"{self.name}_{self.version}",
                name=self.name,
                version=self.version,
                author="Unknown",
                description=f"Model {self.name} version {self.version}"
            )

    def mark_fitted(self) -> None:
        """Marca el modelo como ajustado."""
        self.state = ModelState.FITTED
        self.updated_at = datetime.now().isoformat()

    def mark_ready(self) -> None:
        """Marca el modelo como listo para producción."""
        if not self.state.can_predict():
            raise ValueError(
                f"Model {self.name} must be fitted before marking as ready. "
                f"Current state: {self.state.name}"
            )
        self.state = ModelState.READY
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, error_message: str = "") -> None:
        """Marca el modelo como fallido."""
        self.state = ModelState.FAILED
        self.updated_at = datetime.now().isoformat()
        if error_message:
            # Podríamos guardar el mensaje en metadata si fuera necesario
            pass

    def check_is_fitted(self) -> None:
        """Verifica si el modelo ha sido ajustado."""
        if not self.state.can_predict():
            raise ValueError(
                f"Model {self.name} is not fitted yet. "
                f"Current state: {self.state.name}"
            )

    @property
    def is_fitted(self) -> bool:
        """Propiedad de compatibilidad para verificar si está ajustado."""
        return self.state.can_predict()

    @property
    def is_ready(self) -> bool:
        """Verifica si el modelo está listo para producción."""
        return self.state == ModelState.READY


@dataclass
class ModelConfig:
    """Configuración base para modelos con validación.

    Esta clase debe ser heredada por configs específicos de modelos.
    Proporciona parámetros comunes y validación básica.

    Example:
        Crear una configuración específica para un modelo::

            @dataclass
            class LogisticRegressionConfig(ModelConfig):
                C: float = 1.0  # Regularization strength
                penalty: str = "l2"
                solver: str = "lbfgs"

                def __post_init__(self):
                    super().__post_init__()
                    if self.C <= 0:
                        raise ValueError("C must be positive")
                    if self.penalty not in ["l1", "l2", "elasticnet", "none"]:
                        raise ValueError(f"Invalid penalty: {self.penalty}")

    Attributes:
        random_state: Semilla para generador de números aleatorios
        verbose: Si True, imprime mensajes de progreso
        n_jobs: Número de trabajos paralelos (-1 para usar todos los cores)
        validation_fraction: Fracción de datos para validación (0.0-1.0)
        early_stopping: Si True, detiene el entrenamiento cuando no hay mejora
        max_iter: Número máximo de iteraciones
        tol: Tolerancia para criterio de convergencia
    """

    random_state: int = 42
    verbose: bool = False
    n_jobs: int = 1
    validation_fraction: float = 0.1
    early_stopping: bool = False
    max_iter: int = 1000
    tol: float = 1e-4

    def __post_init__(self) -> None:
        """Validación de configuración base."""
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}")

        if not 0 <= self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}"
            )

        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")

        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError(
                f"n_jobs must be -1 (all cores) or positive, got {self.n_jobs}"
            )


