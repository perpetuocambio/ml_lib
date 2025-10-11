"""AI message abstraction for LLM clients."""

from __future__ import annotations

from ml_lib.llm.messages.base_message import BaseMessage

try:
    from langchain_core.messages import AIMessage as LangChainAIMessage

    # Use LangChain's AIMessage when available
    AIMessage = LangChainAIMessage

except ImportError:
    # Fallback implementation for testing environments
    class AIMessage(BaseMessage):
        """AI-generated message."""

        def __init__(self, content: str):
            super().__init__(content)

        def get_type(self) -> str:
            """Get the message type."""
            return "ai"
