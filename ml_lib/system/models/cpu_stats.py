from dataclasses import dataclass
from typing import Optional


@dataclass
class CPUStats:
    """CPU statistics."""

    usage_percent: float
    temperature_celsius: Optional[float] = None
    frequency_mhz: Optional[float] = None
    core_count: int = 0

    def is_thermal_throttling(self, threshold: float = 85.0) -> bool:
        """Check if CPU is thermal throttling."""
        return (
            self.temperature_celsius is not None
            and self.temperature_celsius >= threshold
        )
