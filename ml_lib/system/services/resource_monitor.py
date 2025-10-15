"""
Resource Monitor - Reutilizable para cualquier proyecto ML.

Monitorea:
- GPU: uso, memoria, temperatura
- CPU: uso, temperatura
- RAM: disponible, usado
- Disco: espacio disponible

Diseñado para ser independiente y reutilizable.
"""

import logging
import time

from ml_lib.system.models.cpu_stats import CPUStats
from ml_lib.system.models.gpu_stats import GPUStats
from ml_lib.system.models.ram_stats import RAMStats
from ml_lib.system.models.system_resources import SystemResources

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    System resource monitor.

    Reutilizable, sin dependencias del resto del proyecto.

    Example:
        >>> monitor = ResourceMonitor()
        >>> stats = monitor.get_current_stats()
        >>> print(f"GPU Memory: {stats.get_primary_gpu().memory_used_gb:.2f}GB")
        >>> print(f"GPU Temp: {stats.get_primary_gpu().temperature_celsius}°C")
        >>> print(f"CPU Usage: {stats.cpu.usage_percent}%")
    """

    def __init__(self, enable_nvidia_smi: bool = True):
        """
        Initialize resource monitor.

        Args:
            enable_nvidia_smi: Use nvidia-smi for detailed GPU stats (slower but more info)
        """
        self.enable_nvidia_smi = enable_nvidia_smi
        self._has_torch = self._check_torch()
        self._has_psutil = self._check_psutil()
        self._has_nvidia_ml = self._check_nvidia_ml()

        logger.info(
            f"ResourceMonitor initialized (torch={self._has_torch}, "
            f"psutil={self._has_psutil}, nvidia-ml={self._has_nvidia_ml})"
        )

    def _check_torch(self) -> bool:
        """Check if torch is available."""
        try:
            import torch

            return True
        except ImportError:
            return False

    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil

            return True
        except ImportError:
            return False

    def _check_nvidia_ml(self) -> bool:
        """Check if pynvml is available."""
        try:
            import pynvml

            return True
        except ImportError:
            return False

    def get_gpu_stats(self) -> list[GPUStats]:
        """
        Get GPU statistics.

        Returns:
            List of GPUStats for each GPU
        """
        gpus = []

        if not self._has_torch:
            return gpus

        try:
            if not torch.cuda.is_available():
                return gpus

            # Use pynvml if available for detailed stats
            if self._has_nvidia_ml and self.enable_nvidia_smi:
                gpus = self._get_gpu_stats_nvidia_ml()
            else:
                # Fallback to torch only
                gpus = self._get_gpu_stats_torch()

        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")

        return gpus

    def _get_gpu_stats_torch(self) -> list[GPUStats]:
        """Get GPU stats using torch only (basic info)."""
        import torch

        gpus = []
        for gpu_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_id)

            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id)
            memory_total = props.total_memory

            gpus.append(
                GPUStats(
                    gpu_id=gpu_id,
                    name=props.name,
                    memory_used_mb=memory_allocated / (1024**2),
                    memory_total_mb=memory_total / (1024**2),
                    memory_free_mb=(memory_total - memory_allocated) / (1024**2),
                    utilization_percent=0.0,  # Not available via torch
                    temperature_celsius=None,  # Not available via torch
                )
            )

        return gpus

    def _get_gpu_stats_nvidia_ml(self) -> list[GPUStats]:
        """Get GPU stats using pynvml (detailed info)."""
        try:
            import pynvml

            pynvml.nvmlInit()

            gpus = []
            device_count = pynvml.nvmlDeviceGetCount()

            for gpu_id in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

                # Basic info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = float(util.gpu)
                except:
                    utilization = 0.0

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    temperature = float(temp)
                except:
                    temperature = None

                # Power
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_watts = power_mw / 1000.0
                except:
                    power_watts = None

                # Fan speed
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                    fan_speed = float(fan)
                except:
                    fan_speed = None

                gpus.append(
                    GPUStats(
                        gpu_id=gpu_id,
                        name=name,
                        memory_used_mb=mem_info.used / (1024**2),
                        memory_total_mb=mem_info.total / (1024**2),
                        memory_free_mb=mem_info.free / (1024**2),
                        utilization_percent=utilization,
                        temperature_celsius=temperature,
                        power_watts=power_watts,
                        fan_speed_percent=fan_speed,
                    )
                )

            pynvml.nvmlShutdown()
            return gpus

        except Exception as e:
            logger.warning(f"pynvml failed, falling back to torch: {e}")
            return self._get_gpu_stats_torch()

    def get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics."""
        if not self._has_psutil:
            return CPUStats(usage_percent=0.0, core_count=0)

        try:
            import psutil

            # CPU usage (brief interval for accuracy)
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # CPU temperature (not always available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if "coretemp" in temps:
                    # Average of all cores
                    core_temps = [t.current for t in temps["coretemp"]]
                    temperature = sum(core_temps) / len(core_temps)
                elif "cpu_thermal" in temps:  # Raspberry Pi
                    temperature = temps["cpu_thermal"][0].current
            except:
                pass

            # CPU frequency
            frequency = None
            try:
                freq = psutil.cpu_freq()
                if freq:
                    frequency = freq.current
            except:
                pass

            return CPUStats(
                usage_percent=cpu_percent,
                temperature_celsius=temperature,
                frequency_mhz=frequency,
                core_count=psutil.cpu_count(logical=False) or 0,
            )

        except Exception as e:
            logger.warning(f"Failed to get CPU stats: {e}")
            return CPUStats(usage_percent=0.0, core_count=0)

    def get_ram_stats(self) -> RAMStats:
        """Get RAM statistics."""
        if not self._has_psutil:
            return RAMStats(total_mb=0, used_mb=0, available_mb=0, percent_used=0.0)

        try:
            import psutil

            mem = psutil.virtual_memory()

            return RAMStats(
                total_mb=mem.total / (1024**2),
                used_mb=mem.used / (1024**2),
                available_mb=mem.available / (1024**2),
                percent_used=mem.percent,
            )

        except Exception as e:
            logger.warning(f"Failed to get RAM stats: {e}")
            return RAMStats(total_mb=0, used_mb=0, available_mb=0, percent_used=0.0)

    def get_current_stats(self) -> SystemResources:
        """
        Get current system resource snapshot.

        Returns:
            SystemResources with all stats

        Example:
            >>> monitor = ResourceMonitor()
            >>> stats = monitor.get_current_stats()
            >>> gpu = stats.get_primary_gpu()
            >>> if gpu:
            ...     print(f"GPU: {gpu.memory_used_gb:.1f}GB / {gpu.memory_total_gb:.1f}GB")
            ...     print(f"Temp: {gpu.temperature_celsius}°C")
        """
        return SystemResources(
            timestamp=time.time(),
            gpus=self.get_gpu_stats(),
            cpu=self.get_cpu_stats(),
            ram=self.get_ram_stats(),
        )

    def can_fit_model(self, estimated_size_gb: float, device: str = "cuda") -> bool:
        """
        Check if a model of given size can fit.

        Args:
            estimated_size_gb: Estimated model size in GB
            device: Target device ("cuda" or "cpu")

        Returns:
            True if model should fit
        """
        stats = self.get_current_stats()

        if device == "cuda":
            if not stats.has_gpu():
                return False
            available = stats.available_gpu_memory_gb()
            # Add 10% buffer
            return available >= (estimated_size_gb * 1.1)
        else:
            # CPU = RAM
            available = stats.ram.available_gb
            return available >= (estimated_size_gb * 1.1)

    def get_recommended_device(self, model_size_gb: float) -> str:
        """
        Get recommended device for model.

        Args:
            model_size_gb: Model size in GB

        Returns:
            "cuda" or "cpu"
        """
        if self.can_fit_model(model_size_gb, "cuda"):
            return "cuda"
        elif self.can_fit_model(model_size_gb, "cpu"):
            return "cpu"
        else:
            logger.warning(f"Model ({model_size_gb}GB) may not fit in available memory")
            return "cpu"  # Fallback

    def print_summary(self):
        """Print human-readable summary."""
        stats = self.get_current_stats()

        print("=" * 60)
        print("System Resources")
        print("=" * 60)

        # GPU
        if stats.has_gpu():
            for gpu in stats.gpus:
                print(f"\n📊 GPU {gpu.gpu_id}: {gpu.name}")
                print(
                    f"   Memory: {gpu.memory_used_gb:.1f}GB / {gpu.memory_total_gb:.1f}GB ({gpu.memory_percent:.1f}%)"
                )
                print(f"   Utilization: {gpu.utilization_percent:.1f}%")
                if gpu.temperature_celsius:
                    emoji = "🔥" if gpu.is_thermal_throttling() else "🌡️"
                    print(f"   {emoji} Temperature: {gpu.temperature_celsius}°C")
                if gpu.power_watts:
                    print(f"   ⚡ Power: {gpu.power_watts:.1f}W")
        else:
            print("\n⚠️  No GPU detected")

        # CPU
        print(f"\n📊 CPU ({stats.cpu.core_count} cores)")
        print(f"   Usage: {stats.cpu.usage_percent:.1f}%")
        if stats.cpu.temperature_celsius:
            emoji = "🔥" if stats.cpu.is_thermal_throttling() else "🌡️"
            print(f"   {emoji} Temperature: {stats.cpu.temperature_celsius}°C")
        if stats.cpu.frequency_mhz:
            print(f"   Frequency: {stats.cpu.frequency_mhz:.0f} MHz")

        # RAM
        print(f"\n📊 RAM")
        print(f"   Total: {stats.ram.total_gb:.1f}GB")
        print(f"   Used: {stats.ram.used_gb:.1f}GB ({stats.ram.percent_used:.1f}%)")
        print(f"   Available: {stats.ram.available_gb:.1f}GB")

        # Warnings
        if stats.any_thermal_issues():
            print("\n⚠️  THERMAL THROTTLING DETECTED")

        print("=" * 60)
