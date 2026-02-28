"""God Mode — omniscient view of all agent activity.

Shows all active conversations, reflections, mood changes,
memory updates, and autonomy decisions simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ui.widgets import ConversationView, EventLogWidget, StatsWidget, WorldStatusWidget

if TYPE_CHECKING:
    from world.orchestrator import Orchestrator


class GodModeScreen(Screen):
    """God Mode: see everything happening in the world."""

    BINDINGS = [
        ("p", "switch_participant", "Participant Mode"),
        ("q", "quit_app", "Quit"),
    ]

    CSS = """
    #god-left {
        width: 30;
        border-right: solid $accent;
    }
    #god-right {
        width: 1fr;
    }
    #god-world-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
    }
    #god-stats-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
        margin-top: 1;
    }
    #god-events-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
        margin-top: 1;
    }
    #god-conv-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
    }
    WorldStatusWidget {
        height: auto;
        padding: 0 1;
    }
    StatsWidget {
        height: auto;
        padding: 0 1;
    }
    EventLogWidget {
        height: 1fr;
        padding: 0 1;
    }
    ConversationView {
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(self, orchestrator: Orchestrator, **kwargs):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="god-left"):
                yield Static("\U0001f30d World Status", id="god-world-title")
                yield WorldStatusWidget(id="god-world-status")
                yield Static("\U0001f4ca Stats", id="god-stats-title")
                yield StatsWidget(id="god-stats")
                yield Static("\U0001f4dc Event Log", id="god-events-title")
                yield EventLogWidget(id="god-events", wrap=True, markup=True)
            with Vertical(id="god-right"):
                yield Static("\U0001f4ac All Activity", id="god-conv-title")
                yield ConversationView(id="god-conversation", wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_world_status()
        self.set_interval(5, self.refresh_world_status)
        self.log_event("God Mode aktif")

    def refresh_world_status(self) -> None:
        try:
            self.query_one("#god-world-status", WorldStatusWidget).update_status(
                self.orchestrator.registry
            )
        except Exception:
            pass

    def log_event(self, text: str, style: str = "") -> None:
        try:
            self.query_one("#god-events", EventLogWidget).add_event(text, style)
        except Exception:
            pass

    def log_conversation(self, speaker: str, message: str, emoji: str = "") -> None:
        try:
            conv = self.query_one("#god-conversation", ConversationView)
            conv.add_agent_message(speaker, message, emoji)
        except Exception:
            pass

    def log_reflection(self, agent_name: str, reflection: str) -> None:
        try:
            conv = self.query_one("#god-conversation", ConversationView)
            conv.add_reflection(agent_name, reflection)
        except Exception:
            pass

    def action_switch_participant(self) -> None:
        self.app.switch_to_participant_mode()

    def action_quit_app(self) -> None:
        self.app.exit()
