"""
Enumeraciones para el módulo core de ml_lib.
"""

from enum import Enum, auto


class ModelState(Enum):
    """Estados del ciclo de vida de un modelo."""
    INITIALIZED = auto()
    TRAINING = auto()
    FITTED = auto()
    VALIDATING = auto()
    READY = auto()
    FAILED = auto()

    def is_ready(self) -> bool:
        """Verifica si el modelo está listo para usar."""
        return self == ModelState.READY

    def is_failed(self) -> bool:
        """Verifica si el modelo ha fallado."""
        return self == ModelState.FAILED

    def can_predict(self) -> bool:
        """Verifica si el modelo puede hacer predicciones."""
        return self in (ModelState.FITTED, ModelState.READY)


class TrainingMode(Enum):
    """Modo de entrenamiento del modelo."""
    BATCH = "batch"
    ONLINE = "online"
    MINI_BATCH = "mini_batch"
    INCREMENTAL = "incremental"


class ValidationStrategy(Enum):
    """Estrategia de validación de datos."""
    HOLDOUT = "holdout"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    LEAVE_ONE_OUT = "leave_one_out"


class ErrorSeverity(Enum):
    """Nivel de severidad de errores."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def is_critical(self) -> bool:
        """Verifica si el error es crítico."""
        return self == ErrorSeverity.CRITICAL

    def requires_immediate_action(self) -> bool:
        """Verifica si requiere acción inmediata."""
        return self in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL)
