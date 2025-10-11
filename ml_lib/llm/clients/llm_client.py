"""LLM client interface for agent reasoning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from ml_lib.llm.messages.ai_message import AIMessage
from ml_lib.llm.messages.base_message import BaseMessage

# SystemMessage and UserMessage may not exist yet - remove these imports if not found


class LLMClient(ABC):
    """Abstract interface for LLM clients used by agents."""

    @abstractmethod
    async def generate(self, messages: list[BaseMessage]) -> AIMessage:
        """Generate a response from the LLM."""

    def validate_messages(self, messages: list[BaseMessage]) -> bool:
        """Validate message list for common issues."""
        if not messages:
            return False

        # Check for empty content
        for message in messages:
            if not message.content or message.content.strip() == "":
                return False

        return True

    def get_message_count(self, messages: list[BaseMessage]) -> int:
        """Get total number of messages."""
        return len(messages)

    def get_total_content_length(self, messages: list[BaseMessage]) -> int:
        """Get total character count across all messages."""
        return sum(len(message.content) for message in messages)

    def get_messages_summary(self, messages: list[BaseMessage]) -> str:
        """Get summary of message types and counts."""
        message_types = Counter(type(message).__name__ for message in messages)
        total_chars = self.get_total_content_length(messages)

        type_summary = ", ".join(
            f"{count} {msg_type}" for msg_type, count in message_types.items()
        )
        return f"Messages: {len(messages)} ({type_summary}), Total chars: {total_chars}"

    def truncate_messages_by_length(
        self, messages: list[BaseMessage], max_length: int
    ) -> list[BaseMessage]:
        """Truncate message list to stay within character limit."""
        if max_length <= 0:
            return messages

        result = []
        current_length = 0

        # Keep messages from the end (most recent) while staying under limit
        for message in reversed(messages):
            message_length = len(message.content)
            if current_length + message_length <= max_length:
                result.insert(0, message)
                current_length += message_length
            else:
                break

        return result

    def get_conversation_context(self, messages: list[BaseMessage]) -> str:
        """Get a brief context summary of the conversation."""
        if not messages:
            return "Empty conversation"

        user_messages = sum(
            1 for msg in messages if type(msg).__name__ == "UserMessage"
        )
        ai_messages = sum(1 for msg in messages if type(msg).__name__ == "AIMessage")
        system_messages = sum(
            1 for msg in messages if type(msg).__name__ == "SystemMessage"
        )

        return f"Conversation: {user_messages} user, {ai_messages} AI, {system_messages} system messages"
