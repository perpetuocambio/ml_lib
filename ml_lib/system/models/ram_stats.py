from dataclasses import dataclass


@dataclass
class RAMStats:
    """RAM statistics."""

    total_mb: float
    used_mb: float
    available_mb: float
    percent_used: float

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024

    @property
    def used_gb(self) -> float:
        return self.used_mb / 1024

    @property
    def available_gb(self) -> float:
        return self.available_mb / 1024
