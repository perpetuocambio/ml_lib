from dataclasses import dataclass
from typing import Optional
from ml_lib.system.models.cpu_stats import CPUStats
from ml_lib.system.models.gpu_stats import GPUStats
from ml_lib.system.models.ram_stats import RAMStats


@dataclass
class SystemResources:
    """Complete system resource snapshot."""

    timestamp: float
    gpus: list[GPUStats]
    cpu: CPUStats
    ram: RAMStats

    def get_primary_gpu(self) -> Optional[GPUStats]:
        """Get primary GPU (GPU 0)."""
        return self.gpus[0] if self.gpus else None

    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return len(self.gpus) > 0

    def total_gpu_memory_gb(self) -> float:
        """Total GPU memory across all GPUs."""
        return sum(gpu.memory_total_gb for gpu in self.gpus)

    def available_gpu_memory_gb(self) -> float:
        """Available GPU memory across all GPUs."""
        return sum(gpu.memory_free_gb for gpu in self.gpus)

    def any_thermal_issues(self) -> bool:
        """Check if any component has thermal issues."""
        if self.cpu.is_thermal_throttling():
            return True
        return any(gpu.is_thermal_throttling() for gpu in self.gpus)
