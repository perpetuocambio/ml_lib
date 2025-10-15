"""Simple Dependency Injection Container.

Lightweight DI implementation for ml_lib.
Supports:
- Singleton registration
- Transient registration
- Factory registration
- Constructor injection
"""

from typing import Any, Callable, Dict, Type, TypeVar, Protocol
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DIContainer:
    """
    Simple Dependency Injection container.

    Example:
        >>> container = DIContainer()
        >>> container.register_singleton(IAnalyzer, ConcreteAnalyzer)
        >>> analyzer = container.resolve(IAnalyzer)
    """

    def __init__(self):
        """Initialize container."""
        self._singletons: Dict[Type, Any] = {}
        self._transients: Dict[Type, Type] = {}
        self._factories: Dict[Type, Callable] = {}
        self._instances: Dict[Type, Any] = {}  # Cached singleton instances

        logger.debug("DIContainer initialized")

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """
        Register a singleton (one instance for lifetime of container).

        Args:
            interface: Interface type
            implementation: Implementation class
        """
        self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__name__}")

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """
        Register a transient (new instance every resolve).

        Args:
            interface: Interface type
            implementation: Implementation class
        """
        self._transients[interface] = implementation
        logger.debug(f"Registered transient: {interface.__name__} -> {implementation.__name__}")

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """
        Register a factory function.

        Args:
            interface: Interface type
            factory: Factory function that returns instance
        """
        self._factories[interface] = factory
        logger.debug(f"Registered factory: {interface.__name__}")

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register an existing instance.

        Args:
            interface: Interface type
            instance: Pre-constructed instance
        """
        self._instances[interface] = instance
        logger.debug(f"Registered instance: {interface.__name__}")

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve an instance of the requested interface.

        Args:
            interface: Interface to resolve

        Returns:
            Instance of the implementation

        Raises:
            ValueError: If interface not registered
        """
        # Check if instance already exists
        if interface in self._instances:
            return self._instances[interface]

        # Check factories
        if interface in self._factories:
            factory = self._factories[interface]
            return factory()

        # Check singletons
        if interface in self._singletons:
            implementation = self._singletons[interface]
            instance = self._create_instance(implementation)
            self._instances[interface] = instance  # Cache
            return instance

        # Check transients
        if interface in self._transients:
            implementation = self._transients[interface]
            return self._create_instance(implementation)

        raise ValueError(
            f"Interface {interface.__name__} not registered in DI container. "
            f"Use container.register_singleton/transient/factory first."
        )

    def _create_instance(self, implementation: Type[T]) -> T:
        """
        Create instance with constructor injection.

        Args:
            implementation: Class to instantiate

        Returns:
            New instance with dependencies injected
        """
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        params = sig.parameters

        # Build kwargs by resolving dependencies
        kwargs = {}
        for param_name, param in params.items():
            if param_name == 'self':
                continue

            # Get parameter type annotation
            param_type = param.annotation

            # Skip if no annotation
            if param_type == inspect.Parameter.empty:
                continue

            # If parameter has default, it's optional (don't inject)
            if param.default != inspect.Parameter.empty:
                # Optional parameter - skip DI, use default
                logger.debug(f"Skipping optional param {param_name} in {implementation.__name__}")
                continue

            # Required dependency - try to resolve
            try:
                kwargs[param_name] = self.resolve(param_type)
            except (ValueError, AttributeError) as e:
                logger.error(
                    f"Failed to resolve required dependency for "
                    f"{implementation.__name__}.{param_name}: {e}"
                )
                raise ValueError(
                    f"Cannot resolve {param_name} for {implementation.__name__}"
                )

        # Create instance
        logger.debug(f"Creating instance of {implementation.__name__}")
        return implementation(**kwargs)

    def clear(self) -> None:
        """Clear all registrations and cached instances."""
        self._singletons.clear()
        self._transients.clear()
        self._factories.clear()
        self._instances.clear()
        logger.debug("Container cleared")


# Global container instance (can be replaced for testing)
_default_container: DIContainer | None = None


def get_container() -> DIContainer:
    """Get the global DI container (creates if doesn't exist)."""
    global _default_container
    if _default_container is None:
        _default_container = DIContainer()
    return _default_container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _default_container
    _default_container = None
