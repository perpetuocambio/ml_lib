"""Adapter registry for managing multiple adapters."""

import logging
from ml_lib.diffusion.registry.adapter_registration import AdapterRegistration
from ml_lib.diffusion.registry.adapter_type import AdapterType


logger = logging.getLogger(__name__)


class AdapterRegistry:
    """
    Central registry for managing all active adapters.

    Handles:
    - Multiple ControlNets + IP-Adapters + LoRAs
    - Priority-based application
    - Conflict resolution
    """

    def __init__(self):
        self.adapters: dict[str, AdapterRegistration] = {}
        logger.info("AdapterRegistry initialized")

    def register(
        self,
        adapter_id: str,
        adapter_type: AdapterType,
        priority: int = 50,
        weight: float = 1.0,
    ) -> None:
        """Register an adapter."""
        self.adapters[adapter_id] = AdapterRegistration(
            adapter_id=adapter_id,
            adapter_type=adapter_type,
            priority=priority,
            weight=weight,
        )
        logger.info(
            f"Registered {adapter_type.value}: {adapter_id} (priority={priority})"
        )

    def unregister(self, adapter_id: str) -> None:
        """Unregister an adapter."""
        if adapter_id in self.adapters:
            del self.adapters[adapter_id]
            logger.info(f"Unregistered adapter: {adapter_id}")

    def get_ordered_adapters(self) -> list[AdapterRegistration]:
        """Get adapters ordered by priority (highest first)."""
        active = [a for a in self.adapters.values() if a.active]
        return sorted(active, key=lambda x: x.priority, reverse=True)

    def clear(self) -> None:
        """Clear all registered adapters."""
        self.adapters.clear()
        logger.info("Cleared all adapters")
