"""Protocol for LLM client."""

from typing import Protocol, Optional, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        ...
