"""
Metadatos y modelos auxiliares con tipado estricto.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class ModelMetadata:
    """Metadatos detallados del modelo."""
    
    model_id: str
    name: str
    version: str
    author: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(init=False)
    last_modified: str = field(init=False)
    license: str = "MIT"
    language: str = "Python"
    
    def __post_init__(self) -> None:
        """Inicialización post-creación."""
        from datetime import datetime
        self.created_at = datetime.now().isoformat()
        self.last_modified = self.created_at


@dataclass
class TrainingHistory:
    """Historial de entrenamiento."""
    
    epoch: int
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(init=False)
    learning_rate: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Inicialización post-creación."""
        from datetime import datetime
        self.timestamp = datetime.now().isoformat()


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del modelo."""
    
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    auc: Optional[float] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_score(self, metric_name: str) -> Optional[float]:
        """Obtiene un valor de métrica por nombre."""
        return getattr(self, metric_name, self.additional_metrics.get(metric_name))
    
    def set_score(self, metric_name: str, value: float) -> None:
        """Establece un valor de métrica."""
        if hasattr(self, metric_name):
            setattr(self, metric_name, value)
        else:
            self.additional_metrics[metric_name] = value