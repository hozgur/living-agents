"""Reusable Textual widgets for the Living Agents TUI."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, RichLog, Static

# Chat transcript log path
CHAT_LOG_PATH = Path("data/chat.log")

# Regex to strip Rich markup tags like [bold green], [/], [dim], etc.
_RICH_TAG_RE = re.compile(r"\[/?[^\]]*\]")

# Regex to strip XML-style tool call blocks that Claude sometimes emits as text
_XML_TOOL_CALL_RE = re.compile(
    r"<function_calls>.*?</function_calls>\s*"
    r"|<function_result>.*?</function_result>\s*"
    r"|<invoke\b[^>]*>.*?</invoke>\s*",
    re.DOTALL,
)


class WorldStatusWidget(Static):
    """Displays current world entity status in the left panel."""

    status_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Label(self.status_text, id="world-status-label")

    def watch_status_text(self, value: str) -> None:
        try:
            self.query_one("#world-status-label", Label).update(value)
        except Exception:
            pass

    def update_status(self, registry) -> None:
        """Refresh from WorldRegistry."""
        entities = registry.get_all()
        if not entities:
            self.status_text = "(Nobody here yet)"
            return

        lines = []
        for entity in entities:
            status_icon = {
                "online": "[green]\u2b24[/]",
                "idle": "[green]\u2b24[/]",
                "in_conversation": "[blue]\u2b24[/]",
                "thinking": "[yellow]\u2b24[/]",
                "reflecting": "[magenta]\u2b24[/]",
                "offline": "[dim]\u2b24[/]",
            }.get(entity.status, "[dim]\u2b24[/]")

            conv_info = ""
            if entity.current_conversation_with:
                conv_info = f" -> {entity.current_conversation_with[:12]}"

            lines.append(
                f"{status_icon} {entity.avatar_emoji} {entity.name} ({entity.status}){conv_info}"
            )

        self.status_text = "\n".join(lines)


class EventLogWidget(RichLog):
    """Scrolling event log."""

    def add_event(self, text: str, style: str = "") -> None:
        """Add a timestamped event entry."""
        now = datetime.now(timezone.utc).strftime("%H:%M")
        if style:
            self.write(f"[{style}][{now}] {text}[/]")
        else:
            self.write(f"[dim][{now}][/] {text}")


class ConversationView(RichLog):
    """Displays conversation messages in a group chat style.

    All messages are also appended to data/chat.log as plain text
    so users can copy/paste and share conversation excerpts.
    """

    # Color palette for different agents — cycles if more agents than colors
    AGENT_COLORS = ["green", "yellow", "cyan", "magenta", "blue", "red"]
    _agent_color_map: dict[str, str] = {}

    def _get_agent_color(self, agent_name: str) -> str:
        """Assign a consistent color to each agent."""
        if agent_name not in self._agent_color_map:
            idx = len(self._agent_color_map) % len(self.AGENT_COLORS)
            self._agent_color_map[agent_name] = self.AGENT_COLORS[idx]
        return self._agent_color_map[agent_name]

    @staticmethod
    def _log_to_file(text: str) -> None:
        """Append a plain-text line to the chat log file."""
        try:
            plain = _RICH_TAG_RE.sub("", text)
            CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(plain + "\n")
        except Exception:
            pass

    def add_user_message(self, sender: str, content: str) -> None:
        now = datetime.now(timezone.utc).strftime("%H:%M")
        line = f"[dim]{now}[/] [bold cyan]🧑 {sender}:[/] {content}"
        self.write(line)
        self._log_to_file(f"{now} 🧑 {sender}: {content}")

    def add_agent_message(self, agent_name: str, content: str, emoji: str = "") -> None:
        now = datetime.now(timezone.utc).strftime("%H:%M")
        color = self._get_agent_color(agent_name)
        prefix = f"{emoji} " if emoji else ""
        # Strip any XML tool call artifacts from agent responses
        clean = _XML_TOOL_CALL_RE.sub("", content).strip()
        if not clean:
            return  # nothing to display after stripping
        self.write(f"[dim]{now}[/] [bold {color}]{prefix}{agent_name}:[/] {clean}")
        self._log_to_file(f"{now} {prefix}{agent_name}: {clean}")

    def add_system_message(self, content: str) -> None:
        self.write(f"[dim italic]{content}[/]")
        # Don't log system messages (UI noise like "düşünüyor..." etc.)

    def add_reflection(self, agent_name: str, reflection: str) -> None:
        now = datetime.now(timezone.utc).strftime("%H:%M")
        self.write(f"[dim]{now}[/] [magenta italic]💭 {agent_name} thinking: {reflection}[/]")
        self._log_to_file(f"{now} 💭 {agent_name} thinking: {reflection}")


class StatsWidget(Static):
    """Displays world statistics and live token usage."""

    stats_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Label(self.stats_text, id="stats-label")

    def watch_stats_text(self, value: str) -> None:
        try:
            self.query_one("#stats-label", Label).update(value)
        except Exception:
            pass

    def update_stats_sync(self, orchestrator) -> None:
        """Refresh stats from orchestrator (sync, for timer use)."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker()
        agent_count = len(orchestrator.agents)

        self.stats_text = (
            f"Agents: {agent_count}\n"
            f"API: {tracker.api_calls} calls\n"
            f"In: {tracker._fmt(tracker.input_tokens)}\n"
            f"Out: {tracker._fmt(tracker.output_tokens)}\n"
            f"Total: {tracker._fmt(tracker.total_tokens)}"
        )

    async def update_stats(self, orchestrator) -> None:
        """Refresh stats from orchestrator."""
        self.update_stats_sync(orchestrator)
