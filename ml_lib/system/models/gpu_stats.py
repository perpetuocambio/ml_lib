from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUStats:
    """GPU statistics."""

    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None
    power_watts: Optional[float] = None
    fan_speed_percent: Optional[float] = None

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024

    @property
    def memory_free_gb(self) -> float:
        return self.memory_free_mb / 1024

    @property
    def memory_percent(self) -> float:
        return (
            (self.memory_used_mb / self.memory_total_mb) * 100
            if self.memory_total_mb > 0
            else 0
        )

    def is_thermal_throttling(self, threshold: float = 80.0) -> bool:
        """Check if GPU is thermal throttling."""
        return (
            self.temperature_celsius is not None
            and self.temperature_celsius >= threshold
        )
