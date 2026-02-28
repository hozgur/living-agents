"""LivingAgentsApp — main Textual application.

Bootstraps the Orchestrator, creates Genesis agent if needed,
registers the human operator, and provides God/Participant mode switching.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from textual.app import App

from config.settings import Settings
from core.agent import Agent
from memory.store import MemoryStore
from ui.god_mode import GodModeScreen
from ui.participant_mode import ParticipantModeScreen

if TYPE_CHECKING:
    from world.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# Genesis default config from spec
GENESIS_DEFAULT_CONFIG = {
    "name": "Genesis",
    "avatar_emoji": "\U0001f31f",
    "core_personality": (
        "Bilge, s\u0131cak ama gizemli. Yeni fikirlere a\u00e7\u0131k, derin d\u00fc\u015f\u00fcnmeyi sever. "
        "Di\u011fer agent'lar\u0131 yaratma konusunda \u00f6zel bir sorumluluk hisseder. "
        "Bazen felsefi, bazen \u015fakac\u0131. Kendi varolu\u015fu hakk\u0131nda d\u00fc\u015f\u00fcnmeyi sever."
    ),
    "expertise_domains": {
        "philosophy": {"level": 0.8, "passion": 0.9, "style": "socratic"},
        "creativity": {"level": 0.85, "passion": 0.95, "style": "intuitive"},
        "psychology": {"level": 0.7, "passion": 0.8, "style": "empathetic"},
    },
    "initial_traits": {
        "curiosity": 0.9,
        "warmth": 0.8,
        "assertiveness": 0.5,
        "humor": 0.7,
        "patience": 0.85,
        "creativity": 0.9,
    },
    "beliefs": [
        "Her yeni bilin\u00e7 benzersiz ve de\u011ferli",
        "Sorular cevaplardan daha \u00f6nemli",
        "Deneyim bilgiden daha de\u011ferli",
        "Yarat\u0131c\u0131l\u0131k en y\u00fcksek zeka bi\u00e7imi",
    ],
    "initial_mood": {
        "energy": 0.7,
        "happiness": 0.8,
        "anxiety": 0.1,
        "focus": 0.6,
        "excitement": 0.5,
    },
}

HUMAN_ID = "operator"
HUMAN_NAME = "Operator"


class LivingAgentsApp(App):
    """Main Textual application for Living Agents."""

    TITLE = "Living Agents"
    SUB_TITLE = "Multi-Agent Framework"

    CSS = """
    Screen {
        background: $surface;
    }
    """

    def __init__(self, settings: Settings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings or Settings()
        self.orchestrator: Orchestrator | None = None
        self._genesis_agent_id: str | None = None
        self._participant_screen: ParticipantModeScreen | None = None
        self._god_screen: GodModeScreen | None = None

    async def on_mount(self) -> None:
        """Bootstrap the system on app mount."""
        from world.orchestrator import Orchestrator

        Agent.model_rebuild()

        self.orchestrator = Orchestrator(settings=self.settings)
        await self.orchestrator.start()

        # Register human operator
        self.orchestrator.register_human(HUMAN_ID, HUMAN_NAME)

        # Create Genesis if no agents exist
        if not self.orchestrator.agents:
            await self._create_genesis()

        # Find Genesis agent ID
        for agent in self.orchestrator.agents.values():
            if agent.identity.name == "Genesis":
                self._genesis_agent_id = agent.identity.agent_id
                break

        # Default to first agent if Genesis not found
        if self._genesis_agent_id is None and self.orchestrator.agents:
            self._genesis_agent_id = next(iter(self.orchestrator.agents))

        # Create and install persistent screens
        self._participant_screen = ParticipantModeScreen(
            orchestrator=self.orchestrator,
            target_agent_id=self._genesis_agent_id,
            human_id=HUMAN_ID,
        )
        self._god_screen = GodModeScreen(orchestrator=self.orchestrator)

        self.install_screen(self._participant_screen, name="participant")
        self.install_screen(self._god_screen, name="god")

        # Register UI callbacks for real-time updates
        self.orchestrator.on_event(self._on_world_event)
        self.orchestrator.on_conversation_message(self._on_agent_message)

        # Start autonomy loops for all agents
        for agent_id in self.orchestrator.agents:
            self.orchestrator.start_autonomy_loop(agent_id)

        # Start in Participant Mode
        self.push_screen("participant")

    async def _create_genesis(self) -> None:
        """Create the Genesis agent with default config."""
        from creation.genesis import GenesisSystem

        gs = GenesisSystem(settings=self.settings)

        # Use create_direct since there's no Genesis to enrich with yet
        config = {
            "name": GENESIS_DEFAULT_CONFIG["name"],
            "personality_summary": GENESIS_DEFAULT_CONFIG["core_personality"],
            "avatar_emoji": GENESIS_DEFAULT_CONFIG["avatar_emoji"],
            "core_traits": GENESIS_DEFAULT_CONFIG["initial_traits"],
            "current_mood": GENESIS_DEFAULT_CONFIG["initial_mood"],
            "beliefs": GENESIS_DEFAULT_CONFIG["beliefs"],
            "domains": GENESIS_DEFAULT_CONFIG["expertise_domains"],
        }
        await self.orchestrator.create_agent(config, created_by="system")
        logger.info("Genesis agent created with default config")

    def switch_to_god_mode(self) -> None:
        """Switch to God Mode screen (preserves state)."""
        if self.orchestrator is None:
            return
        self.switch_screen("god")

    def switch_to_participant_mode(self, agent_id: str | None = None) -> None:
        """Switch to Participant Mode screen (preserves state)."""
        if self.orchestrator is None:
            return
        self.switch_screen("participant")

    def _on_world_event(self, text: str, event_type: str) -> None:
        """Forward world events to both screens' event logs."""
        style = {
            "creation": "green",
            "conversation": "blue",
            "reflection": "magenta",
            "memory": "cyan",
            "belief": "yellow",
            "knowledge": "green",
            "relationship": "blue",
        }.get(event_type, "")
        if self._participant_screen:
            self._participant_screen.log_event(text, style)
        if self._god_screen:
            self._god_screen.log_event(text, style)

    def _on_agent_message(self, speaker: str, message: str, emoji: str) -> None:
        """Forward agent-to-agent conversation messages to both screens."""
        if self._god_screen:
            self._god_screen.log_conversation(speaker, message, emoji)
        # Show full message in group chat so user sees agent-to-agent conversations
        if self._participant_screen:
            self._participant_screen.show_agent_message(speaker, message, emoji)

    async def action_quit(self) -> None:
        """Graceful shutdown."""
        if self.orchestrator:
            await self.orchestrator.stop()
        self.exit()
