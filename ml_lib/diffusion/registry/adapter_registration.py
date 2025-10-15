from dataclasses import dataclass

from ml_lib.diffusion.registry.adapter_type import AdapterType


@dataclass
class AdapterRegistration:
    """Registration entry for an adapter."""

    adapter_id: str
    adapter_type: AdapterType
    priority: int  # Higher = applied first
    weight: float
    active: bool = True
