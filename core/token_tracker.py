"""Global token usage tracker for Claude API calls."""

from __future__ import annotations

import threading


class TokenTracker:
    """Thread-safe singleton tracking cumulative token usage."""

    _instance: TokenTracker | None = None
    _lock = threading.Lock()

    def __new__(cls) -> TokenTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.api_calls: int = 0

    def record(self, usage) -> None:
        """Record token usage from a Claude API response.usage object."""
        if usage is None:
            return
        self.input_tokens += getattr(usage, "input_tokens", 0)
        self.output_tokens += getattr(usage, "output_tokens", 0)
        self.api_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return (
            f"API: {self.api_calls} | "
            f"In: {self._fmt(self.input_tokens)} | "
            f"Out: {self._fmt(self.output_tokens)} | "
            f"Toplam: {self._fmt(self.total_tokens)}"
        )

    @staticmethod
    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)
