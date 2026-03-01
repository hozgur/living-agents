"""Debug Mode — full-screen event monitor and agent state inspector.

Shows all system events (memory, beliefs, reflections, relationships,
knowledge) in a wide event stream, plus live agent state details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ui.widgets import EventLogWidget, StatsWidget, WorldStatusWidget

if TYPE_CHECKING:
    from world.orchestrator import Orchestrator


class GodModeScreen(Screen):
    """Debug Mode: full event stream + agent state inspector."""

    BINDINGS = [
        ("p", "switch_participant", "Chat Mode"),
        ("q", "quit_app", "Quit"),
    ]

    CSS = """
    #debug-left {
        width: 30;
        border-right: solid $accent;
    }
    #debug-right {
        width: 1fr;
    }
    #debug-world-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
    }
    #debug-stats-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
        margin-top: 1;
    }
    #debug-agents-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
        margin-top: 1;
    }
    #debug-events-title {
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
    #debug-agent-details {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }
    EventLogWidget {
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
            with Vertical(id="debug-left"):
                yield Static("\U0001f30d World", id="debug-world-title")
                yield WorldStatusWidget(id="debug-world-status")
                yield Static("\U0001f4ca Statistics", id="debug-stats-title")
                yield StatsWidget(id="debug-stats")
                yield Static("\U0001f9e0 Agent States", id="debug-agents-title")
                yield Static("(Loading...)", id="debug-agent-details")
            with Vertical(id="debug-right"):
                yield Static(
                    "\U0001f4dc Event Stream (memory, belief, reflection, knowledge)",
                    id="debug-events-title",
                )
                yield EventLogWidget(id="debug-events", wrap=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_world_status()
        self.refresh_agent_details()
        self.refresh_stats()
        self.set_interval(5, self.refresh_world_status)
        self.set_interval(5, self.refresh_stats)
        self.set_interval(10, self.refresh_agent_details)
        self.log_event("Debug Mode active", "green")

    def refresh_world_status(self) -> None:
        try:
            self.query_one("#debug-world-status", WorldStatusWidget).update_status(
                self.orchestrator.registry
            )
        except Exception:
            pass

    def refresh_stats(self) -> None:
        try:
            self.query_one("#debug-stats", StatsWidget).update_stats_sync(
                self.orchestrator
            )
        except Exception:
            pass

    def refresh_agent_details(self) -> None:
        """Build a live summary of all agents' internal state."""
        try:
            lines = []
            for agent in self.orchestrator.agents.values():
                name = agent.identity.name
                emoji = agent.identity.avatar_emoji

                # Traits — show top 3
                traits = agent.character.core_traits
                top = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
                trait_str = ", ".join(f"{k}={v:.2f}" for k, v in top)

                # Mood — highlight notable
                mood = agent.character.current_mood
                notable_mood = [
                    f"{k}={v:.1f}" for k, v in mood.items()
                    if v >= 0.7 or v <= 0.3
                ]
                mood_str = ", ".join(notable_mood) if notable_mood else "normal"

                # Beliefs count + strongest
                beliefs = agent.character.beliefs
                belief_str = f"{len(beliefs)} beliefs"
                if beliefs:
                    strongest = max(beliefs, key=lambda b: b.conviction)
                    belief_str += f" (strongest: {strongest.conviction:.1f})"

                # Relationships
                rels = agent.character.relationships
                rel_str = f"{len(rels)} relationships"

                # Memory
                engine = self.orchestrator.conversation_engines.get(
                    agent.identity.agent_id
                )
                turns = engine.turn_count if engine else 0

                lines.append(
                    f"{emoji} [bold]{name}[/]\n"
                    f"  traits: {trait_str}\n"
                    f"  mood: {mood_str}\n"
                    f"  {belief_str}, {rel_str}\n"
                    f"  turns: {turns}"
                )

            details = "\n\n".join(lines) if lines else "(No agents)"
            self.query_one("#debug-agent-details", Static).update(details)
        except Exception:
            pass

    def log_event(self, text: str, style: str = "") -> None:
        try:
            self.query_one("#debug-events", EventLogWidget).add_event(text, style)
        except Exception:
            pass

    # Keep these for backward compat with terminal_app routing
    def log_conversation(self, speaker: str, message: str, emoji: str = "") -> None:
        """Show agent messages as events in debug mode."""
        prefix = f"{emoji} " if emoji else ""
        self.log_event(f"{prefix}{speaker}: {message[:120]}", "blue")

    def log_reflection(self, agent_name: str, reflection: str) -> None:
        self.log_event(f"💭 {agent_name}: {reflection}", "magenta")

    def action_switch_participant(self) -> None:
        self.app.switch_to_participant_mode()

    def action_quit_app(self) -> None:
        self.app.exit()
