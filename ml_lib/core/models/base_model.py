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
    """Configuración base para modelos."""

    random_state: int = 42
    verbose: bool = False
    n_jobs: int = 1
    validation_fraction: float = 0.1
    early_stopping: bool = False
    max_iter: int = 1000
    tol: float = 1e-4


@dataclass
class Hyperparameters:
    """Contenedor para hiperparámetros con validación."""

    values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida los hiperparámetros después de la inicialización."""
        self._validate_values()

    def _validate_values(self) -> None:
        """Valida los valores de los hiperparámetros."""
        for key, value in self.values.items():
            self._validate_single_param(key, value)

    def _validate_single_param(self, key: str, value: Any) -> None:
        """Valida un solo parámetro."""
        if isinstance(value, (int, float)) and np.isnan(value):
            raise ValueError(f"Hyperparameter {key} cannot be NaN")
        if isinstance(value, (int, float)) and np.isinf(value):
            raise ValueError(f"Hyperparameter {key} cannot be infinite")

    def update(self, **kwargs: Any) -> "Hyperparameters":
        """Actualiza los hiperparámetros."""
        self.values.update(kwargs)
        self._validate_values()
        return self
