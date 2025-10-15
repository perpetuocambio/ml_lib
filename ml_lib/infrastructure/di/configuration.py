"""DI Container Configuration - Composition Root.

This is where all the wiring happens. This file knows about ALL layers
and connects interfaces with implementations.

Following the Dependency Inversion Principle:
- Domain defines interfaces
- Infrastructure implements them
- This file wires them together
"""

from ml_lib.infrastructure.di.container import DIContainer

# Domain interfaces
from ml_lib.diffusion.domain.interfaces.resource_monitor import IResourceMonitor
from ml_lib.diffusion.domain.interfaces.model_registry import IModelRegistry
from ml_lib.diffusion.domain.interfaces.prompt_analyzer import IPromptAnalyzer

# Infrastructure implementations
from ml_lib.infrastructure.monitoring.resource_monitor_adapter import (
    ResourceMonitorAdapter,
)

# Domain services
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
    ILoRARepository,
)

# Application use cases
from ml_lib.diffusion.application.use_cases.generate_image import (
    GenerateImageUseCase,
)


def configure_container() -> DIContainer:
    """
    Configure the DI container with all registrations.

    This is the Composition Root - the only place that knows about
    all layers and how to wire them together.

    Returns:
        Configured DI container
    """
    container = DIContainer()

    # === INFRASTRUCTURE LAYER ===

    # Resource monitoring
    container.register_singleton(IResourceMonitor, ResourceMonitorAdapter)

    # TODO: Add more infrastructure registrations as we migrate:
    # - IModelRegistry -> SQLiteModelRegistry
    # - IPromptAnalyzer -> OllamaPromptAnalyzer or RuleBasedAnalyzer
    # - ILoRARepository -> SQLiteLoRARepository
    # - IDiffusionBackend -> DiffusersBackend or ComfyUIBackend

    # === DOMAIN SERVICES ===

    # LoRA recommendation service
    # NOTE: Currently depends on ILoRARepository which needs implementation
    # container.register_transient(LoRARecommendationService, LoRARecommendationService)

    # === APPLICATION LAYER ===

    # Use cases
    # NOTE: Will be enabled when all dependencies are registered
    # container.register_transient(GenerateImageUseCase, GenerateImageUseCase)

    return container


# Global container instance for convenience
_container: DIContainer | None = None


def get_configured_container() -> DIContainer:
    """
    Get the globally configured container.

    Lazy initialization - container is configured on first access.

    Returns:
        Configured container
    """
    global _container
    if _container is None:
        _container = configure_container()
    return _container


def reset_configuration():
    """
    Reset the global container.

    Useful for testing or reconfiguration.
    """
    global _container
    _container = None


def resolve_use_case(use_case_class: type):
    """
    Convenience function to resolve a use case.

    Args:
        use_case_class: Use case class to resolve

    Returns:
        Instance of use case with all dependencies injected
    """
    container = get_configured_container()
    return container.resolve(use_case_class)
