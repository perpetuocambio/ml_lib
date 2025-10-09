"""
Modelo base con metadatos comunes y tipado estricto.
"""

from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np


@dataclass
class BaseModel:
    """Modelo base con metadatos comunes."""

    name: str
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_fitted: bool = False
    created_at: str = field(init=False)

    def __post_init__(self) -> None:
        """Inicialización post-creación."""
        from datetime import datetime

        self.created_at = datetime.now().isoformat()

    def mark_fitted(self) -> None:
        """Marca el modelo como ajustado."""
        self.is_fitted = True

    def check_is_fitted(self) -> None:
        """Verifica si el modelo ha sido ajustado."""
        if not self.is_fitted:
            raise ValueError(f"Model {self.name} is not fitted yet")


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
