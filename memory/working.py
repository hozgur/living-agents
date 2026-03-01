"""Working memory — active conversation context with auto-compression."""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

COMPRESSION_THRESHOLD = 0.8  # Compress at 80% capacity


class WorkingMemory:
    """Holds active conversation context with token-aware compression."""

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.messages: list[dict[str, str]] = []
        self.summary: str = ""
        self.token_count: int = 0

    def add_message(self, role: str, content: str) -> None:
        """Append a message and update token count."""
        self.messages.append({"role": role, "content": content})
        self.token_count += self.estimate_tokens(content)

    def get_context(self) -> dict[str, Any]:
        """Return summary + messages for prompt building."""
        return {
            "summary": self.summary,
            "messages": list(self.messages),
            "token_count": self.token_count,
        }

    async def compress_if_needed(self, claude_client: Callable) -> bool:
        """Compress old messages if token count exceeds 80% capacity.

        Args:
            claude_client: An async callable that takes a prompt string and
                returns a summary string. This keeps WorkingMemory testable
                without a real Claude API client.

        Returns:
            True if compression occurred, False otherwise.
        """
        if self.token_count < self.max_tokens * COMPRESSION_THRESHOLD:
            return False

        if len(self.messages) < 4:
            return False

        # Take the first half of messages to compress
        split_point = len(self.messages) // 2
        to_compress = self.messages[:split_point]
        remaining = self.messages[split_point:]

        # Build text from messages to compress
        text_parts = []
        for msg in to_compress:
            text_parts.append(f"{msg['role']}: {msg['content']}")
        text_to_compress = "\n".join(text_parts)

        prompt = (
            "Summarize the following conversation excerpt briefly and concisely. "
            "Preserve important information, decisions, and emotional tones:\n\n"
            f"{text_to_compress}"
        )

        try:
            new_summary = await claude_client(prompt)
        except Exception:
            logger.warning("Compression failed, keeping messages as-is")
            return False

        # Update summary
        if self.summary:
            self.summary = f"{self.summary}\n\n{new_summary}"
        else:
            self.summary = new_summary

        # Remove compressed messages and recalculate tokens
        self.messages = remaining
        self.token_count = self.estimate_tokens(self.summary) + sum(
            self.estimate_tokens(m["content"]) for m in self.messages
        )

        logger.debug(
            "Working memory compressed: %d messages removed, token count now %d",
            split_point,
            self.token_count,
        )
        return True

    def clear(self) -> None:
        """Reset for a new conversation."""
        self.messages.clear()
        self.summary = ""
        self.token_count = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: word count * 1.3."""
        return int(len(text.split()) * 1.3)
