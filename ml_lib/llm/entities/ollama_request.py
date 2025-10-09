from __future__ import annotations

from dataclasses import dataclass

from infrastructure.providers.llm.entities.ollama_options import OllamaOptions
from infrastructure.providers.llm.entities.ollama_request_data import OllamaRequestData


@dataclass
class OllamaRequest:
    """Request object for Ollama LLM API calls with model, prompt and options."""

    model: str
    prompt: str
    stream: bool
    options: OllamaOptions
    images: list[str] | None = None

    def to_request_data(self) -> OllamaRequestData:
        """Convert to typed request data."""
        return OllamaRequestData(
            model=self.model,
            prompt=self.prompt,
            stream=self.stream,
            temperature=self.options.temperature,
            num_ctx=self.options.num_ctx,
            images=self.images,
        )
