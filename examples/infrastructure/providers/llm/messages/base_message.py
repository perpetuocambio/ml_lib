"""Base message abstraction for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

try:
    from langchain_core.messages import BaseMessage as LangChainBaseMessage

    # Use LangChain's BaseMessage when available
    BaseMessage = LangChainBaseMessage

except ImportError:
    # Fallback implementation for testing environments
    class BaseMessage(ABC):
        """Abstract base message for LLM communication."""

        def __init__(self, content: str):
            self.content = content

        @abstractmethod
        def get_type(self) -> str:
            """Get the message type."""
