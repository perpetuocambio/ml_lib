from typing_extensions import Protocol


class PipelineProtocol(Protocol):
    """Protocol for diffusion pipeline objects."""

    def enable_sequential_cpu_offload(self) -> None: ...

    def enable_model_cpu_offload(self) -> None: ...
