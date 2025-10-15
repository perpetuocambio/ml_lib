from typing import Protocol


class VAEProtocol(Protocol):
    """Protocol for VAE component."""

    def enable_tiling(self) -> None: ...

    def enable_slicing(self) -> None: ...

    def enable_layerwise_casting(
        self, storage_dtype: object, compute_dtype: object
    ) -> None: ...
