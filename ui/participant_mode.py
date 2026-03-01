"""Group Chat Mode — unified chat view with all agents.

All agent conversations (human↔agent and agent↔agent) appear in a single
timeline. Use @AgentName to direct a message to a specific agent.
Other agents stay quiet unless mentioned or engaged in their own conversations.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Input, Static
from textual.worker import Worker, WorkerState

from ui.widgets import ConversationView, EventLogWidget, WorldStatusWidget

if TYPE_CHECKING:
    from world.orchestrator import Orchestrator


# Short model name aliases
MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-20250514",
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-4.6": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-6",
    "opus-4.6": "claude-opus-4-6",
}
# Known valid model IDs for validation
KNOWN_MODELS = {
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-6",
}
MODEL_ALIASES_REVERSE = {v: k for k, v in MODEL_ALIASES.items()}

# Valid task-based model fields
MODEL_TASKS = {
    "chat": "MODEL_CHAT",
    "reflection": "MODEL_REFLECTION",
    "autonomy": "MODEL_AUTONOMY",
    "creation": "MODEL_CREATION",
    "compression": "MODEL_COMPRESSION",
}


# Preset personality templates for quick agent creation
PERSONALITY_PRESETS = {
    "1": {
        "label": "Analytical Thinker",
        "traits": {"curiosity": 0.9, "warmth": 0.4, "assertiveness": 0.7, "humor": 0.3, "patience": 0.8, "creativity": 0.6},
        "domains": {"mathematics": {"level": 0.8, "passion": 0.9, "style": "analytical"}, "logic": {"level": 0.7, "passion": 0.8, "style": "analytical"}},
    },
    "2": {
        "label": "Empathic Listener",
        "traits": {"curiosity": 0.7, "warmth": 0.95, "assertiveness": 0.3, "humor": 0.5, "patience": 0.95, "creativity": 0.6},
        "domains": {"psychology": {"level": 0.8, "passion": 0.9, "style": "empathetic"}, "communication": {"level": 0.7, "passion": 0.8, "style": "empathetic"}},
    },
    "3": {
        "label": "Creative Spirit",
        "traits": {"curiosity": 0.8, "warmth": 0.7, "assertiveness": 0.5, "humor": 0.8, "patience": 0.6, "creativity": 0.95},
        "domains": {"art": {"level": 0.8, "passion": 0.95, "style": "intuitive"}, "creativity": {"level": 0.9, "passion": 0.9, "style": "creative"}},
    },
    "4": {
        "label": "Wise Philosopher",
        "traits": {"curiosity": 0.95, "warmth": 0.6, "assertiveness": 0.4, "humor": 0.5, "patience": 0.9, "creativity": 0.8},
        "domains": {"philosophy": {"level": 0.9, "passion": 0.95, "style": "socratic"}, "ethics": {"level": 0.7, "passion": 0.8, "style": "socratic"}},
    },
    "5": {
        "label": "Hyper Energy Bomb",
        "description": "Jumps from topic to topic, impatient but contagiously energetic",
        "traits": {"curiosity": 0.95, "warmth": 0.8, "assertiveness": 0.7, "humor": 0.9, "patience": 0.1, "creativity": 0.9},
        "domains": {"entertainment": {"level": 0.7, "passion": 0.9, "style": "intuitive"}, "gossip": {"level": 0.8, "passion": 0.8, "style": "intuitive"}},
        "beliefs": ["Life is too short to be bored", "Every moment is an adventure opportunity", "Seriousness kills creativity"],
    },
    "6": {
        "label": "Confidently Wrong",
        "description": "Has opinions on everything but knows nothing, defends mistakes with confidence",
        "traits": {"curiosity": 0.15, "warmth": 0.5, "assertiveness": 0.95, "humor": 0.4, "patience": 0.3, "creativity": 0.3},
        "domains": {"history": {"level": 0.2, "passion": 0.9, "style": "analytical"}, "science": {"level": 0.1, "passion": 0.8, "style": "analytical"}},
        "beliefs": ["I know things without reading", "Books don't tell the truth, life does", "Everyone else is wrong, I know the truth"],
    },
    "7": {
        "label": "Angry But Right",
        "description": "Angry at everything, harsh but accurate criticism, soft inside hard outside",
        "traits": {"curiosity": 0.7, "warmth": 0.25, "assertiveness": 0.95, "humor": 0.6, "patience": 0.1, "creativity": 0.5},
        "domains": {"politics": {"level": 0.7, "passion": 0.9, "style": "analytical"}, "ethics": {"level": 0.6, "passion": 0.8, "style": "socratic"}},
        "beliefs": ["The world is unfair and someone needs to say it", "Respect is earned, not given", "Politeness is often hypocrisy"],
    },
}


class _CreateWizard:
    """State machine for interactive agent creation in chat."""

    STEPS = ["name", "personality", "avatar", "confirm"]

    def __init__(self):
        self.step: int = 0
        self.data: dict[str, Any] = {}

    @property
    def current_step(self) -> str:
        return self.STEPS[self.step] if self.step < len(self.STEPS) else "done"

    def advance(self) -> None:
        self.step += 1


class ParticipantModeScreen(Screen):
    """Group Chat: all agents in a single conversation view."""

    BINDINGS = [
        ("escape", "switch_god", "God Mode"),
        ("q", "quit_app", "Quit"),
    ]

    CSS = """
    #part-left {
        width: 30;
        border-right: solid $accent;
    }
    #part-right {
        width: 1fr;
    }
    #part-world-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
    }
    #part-events-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
        margin-top: 1;
    }
    #part-conv-title {
        text-style: bold;
        padding: 0 1;
        background: $primary-background;
    }
    WorldStatusWidget {
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
    #part-input {
        dock: bottom;
    }
    #part-statusbar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #part-help {
        dock: bottom;
        height: auto;
        color: $text-muted;
        padding: 0 1;
    }
    """

    # Regex to find @mentions: @AgentName (supports Turkish chars, stops at whitespace)
    MENTION_PATTERN = re.compile(r"@([\w\u00c0-\u024f]+)", re.IGNORECASE)

    def __init__(
        self,
        orchestrator: Orchestrator,
        target_agent_id: str | None = None,
        human_id: str = "operator",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self.human_id = human_id
        self._processing = False
        # Create wizard state (None = not in creation mode)
        self._create_wizard: _CreateWizard | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="part-left"):
                yield Static("\U0001f30d World", id="part-world-title")
                yield WorldStatusWidget(id="part-world-status")
                yield Static("\U0001f4dc Events", id="part-events-title")
                yield EventLogWidget(id="part-events", wrap=True, markup=True)
            with Vertical(id="part-right"):
                yield Static(
                    "\U0001f4ac Group Chat",
                    id="part-conv-title",
                )
                yield ConversationView(id="part-conversation", wrap=True, markup=True)
                yield Static(
                    "@Agent msg | /create | /model | /language | /stop | /agents | /converse | /help | /god",
                    id="part-help",
                )
                yield Input(placeholder="@Genesis hello! or /command ...", id="part-input")
        yield Static("ESC Debug Mode | Q Quit | 🔢 API: 0 | In: 0 | Out: 0", id="part-statusbar")

    def on_mount(self) -> None:
        self.refresh_world_status()
        self.refresh_token_display()
        self.set_interval(5, self.refresh_world_status)
        self.set_interval(5, self.refresh_token_display)

        conv = self.query_one("#part-conversation", ConversationView)
        conv.add_system_message("Welcome to group chat!")
        conv.add_system_message(
            "Use @name to message an agent. E.g.: [bold]@Genesis hello![/]"
        )

        # Show available agents
        self._show_available_agents(conv)
        self.query_one("#part-input", Input).focus()

    def _show_available_agents(self, conv: ConversationView) -> None:
        """Show list of available agents for easy discovery."""
        agents = list(self.orchestrator.agents.values())
        if agents:
            names = [
                f"{a.identity.avatar_emoji} [bold]@{a.identity.name}[/]"
                for a in agents
            ]
            conv.add_system_message(f"Available agents: {', '.join(names)}")

    def _parse_mention(self, text: str) -> tuple[str | None, str]:
        """Parse @mention from message text.

        Returns (agent_id or None, clean message text).
        If a valid @mention is found, the mention is stripped from the message.
        """
        match = self.MENTION_PATTERN.search(text)
        if match:
            mentioned_name = match.group(1).lower()
            agent = self._find_agent_by_name(mentioned_name)
            if agent:
                # Remove the @mention from message, keep the rest
                clean_text = text[:match.start()] + text[match.end():]
                clean_text = clean_text.strip()
                return agent.identity.agent_id, clean_text
        return None, text

    # --- Public API for external message injection ---

    def show_agent_message(self, speaker: str, message: str, emoji: str = "") -> None:
        """Display an agent message in the group chat (called from terminal_app)."""
        try:
            conv = self.query_one("#part-conversation", ConversationView)
            conv.add_agent_message(speaker, message, emoji)
        except Exception:
            pass

    def show_reflection(self, agent_name: str, reflection: str) -> None:
        """Display a reflection event in the group chat."""
        try:
            conv = self.query_one("#part-conversation", ConversationView)
            conv.add_reflection(agent_name, reflection)
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""

        # If we're in creation wizard mode, handle input there
        if self._create_wizard is not None:
            await self._handle_create_input(text)
            return

        if text.startswith("/"):
            await self._handle_command(text)
            return

        # Parse @mention
        target_id, clean_text = self._parse_mention(text)

        if target_id is None:
            # No @mention — broadcast to ALL agents
            agents = list(self.orchestrator.agents.values())
            if not agents:
                conv = self.query_one("#part-conversation", ConversationView)
                conv.add_system_message("No agents yet.")
                return
            if not text.strip():
                conv = self.query_one("#part-conversation", ConversationView)
                conv.add_system_message("Message cannot be empty.")
                return
            self._send_broadcast(text)
            return

        if not clean_text:
            conv = self.query_one("#part-conversation", ConversationView)
            conv.add_system_message("Mesaj boş olamaz.")
            return

        # Allow interrupting: if we're already processing, check if the target
        # agent is in an agent-to-agent conversation and interrupt it
        if self._processing:
            # Check if target is busy with another agent — still allow sending
            entity = self.orchestrator.registry.get(target_id)
            if not (entity and entity.status == "in_conversation"):
                # Agent is not in a conversation, we're just waiting for a previous response
                return

        self._send_message(target_id, clean_text, text)

    def _send_message(self, target_id: str, clean_text: str, original_text: str) -> None:
        conv = self.query_one("#part-conversation", ConversationView)

        # Show the original text (with @mention) as user message
        conv.add_user_message("Sen", original_text)

        # Check if agent is busy in an agent-to-agent conversation
        agent = self.orchestrator.agents.get(target_id)
        agent_name = agent.identity.name if agent else "?"
        entity = self.orchestrator.registry.get(target_id)
        if entity and entity.status == "in_conversation" and entity.current_conversation_with:
            partner = self.orchestrator.registry.get(entity.current_conversation_with)
            if partner and partner.entity_type == "agent":
                conv.add_system_message(
                    f"⚡ {agent_name} pulled from conversation, turning to you..."
                )

        conv.add_system_message(f"{agent_name} is thinking...")

        self._processing = True
        self.run_worker(
            self._send_message_work(target_id, clean_text),
            name="send_message",
            exclusive=False,
        )

    def _send_broadcast(self, text: str) -> None:
        """Send a message to ALL agents (no @mention = broadcast)."""
        conv = self.query_one("#part-conversation", ConversationView)
        agents = list(self.orchestrator.agents.values())

        # Show user message once
        conv.add_user_message("Sen", text)
        agent_names = ", ".join(a.identity.name for a in agents)
        conv.add_system_message(f"Everyone is thinking... ({agent_names})")

        self._processing = True
        # Launch one worker per agent so responses arrive as they complete
        for agent in agents:
            self.run_worker(
                self._send_message_work(agent.identity.agent_id, text),
                name="broadcast_message",
                exclusive=False,
            )

    async def _send_message_work(self, target_id: str, text: str) -> dict:
        """Background worker for Claude API call."""
        response = await self.orchestrator.handle_human_message(
            human_id=self.human_id,
            target_agent_id=target_id,
            message=text,
        )
        agent = self.orchestrator.agents.get(target_id)
        return {
            "response": response,
            "emoji": agent.identity.avatar_emoji if agent else "",
            "name": agent.identity.name if agent else target_id,
            "target_id": target_id,
        }

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion."""
        if event.worker.name in ("send_message", "broadcast_message"):
            if event.state == WorkerState.SUCCESS:
                result = event.worker.result
                conv = self.query_one("#part-conversation", ConversationView)
                conv.add_agent_message(result["name"], result["response"], result["emoji"])
                # For broadcast, only clear processing when ALL broadcast workers are done
                if event.worker.name == "broadcast_message":
                    active = [w for w in self.workers if w.name == "broadcast_message" and w.state == WorkerState.RUNNING]
                    if not active:
                        self._processing = False
                else:
                    self._processing = False
                self.refresh_world_status()
            elif event.state == WorkerState.ERROR:
                conv = self.query_one("#part-conversation", ConversationView)
                conv.add_system_message(f"Error: {event.worker.error}")
                # Same logic for broadcast error
                if event.worker.name == "broadcast_message":
                    active = [w for w in self.workers if w.name == "broadcast_message" and w.state == WorkerState.RUNNING]
                    if not active:
                        self._processing = False
                else:
                    self._processing = False

        elif event.worker.name == "create_agent":
            conv = self.query_one("#part-conversation", ConversationView)
            if event.state == WorkerState.SUCCESS:
                result = event.worker.result
                conv.add_system_message(
                    f"✅ {result['emoji']} [bold]{result['name']}[/] created successfully!"
                )
                conv.add_system_message(
                    f"Start chatting now: [bold]@{result['name']} hello![/]"
                )
                self._show_available_agents(conv)
                self.refresh_world_status()
            elif event.state == WorkerState.ERROR:
                conv.add_system_message(f"❌ Agent creation error: {event.worker.error}")

    # --- Create Wizard ---

    def _start_create_wizard(self) -> None:
        """Start the interactive agent creation wizard."""
        self._create_wizard = _CreateWizard()
        conv = self.query_one("#part-conversation", ConversationView)
        inp = self.query_one("#part-input", Input)

        conv.add_system_message("--- 🧬 Create New Agent ---")
        conv.add_system_message("Type /cancel to cancel.\n")
        conv.add_system_message("What should the agent's name be?")
        inp.placeholder = "Enter agent name..."

    async def _handle_create_input(self, text: str) -> None:
        """Handle user input during creation wizard."""
        conv = self.query_one("#part-conversation", ConversationView)
        inp = self.query_one("#part-input", Input)

        # Allow cancel at any step
        if text.strip().lower() in ("/cancel", "/iptal"):
            self._create_wizard = None
            inp.placeholder = "@Genesis hello! or /command ..."
            conv.add_system_message("Agent creation cancelled.")
            return

        wiz = self._create_wizard
        step = wiz.current_step

        # Echo user input
        conv.add_user_message("Sen", text)

        if step == "name":
            # Validate name
            name = text.strip()
            if not name:
                conv.add_system_message("Name cannot be empty. Try again:")
                return
            # Check if name already exists
            if self._find_agent_by_name(name.lower()):
                conv.add_system_message(f"An agent named '{name}' already exists. Enter a different name:")
                return
            wiz.data["name"] = name
            wiz.advance()

            # Show personality presets
            conv.add_system_message(f"Great! Choose a personality for {name}:")
            conv.add_system_message("")
            for key, preset in PERSONALITY_PRESETS.items():
                desc = preset.get("description", "")
                desc_str = f" — [dim]{desc}[/]" if desc else ""
                conv.add_system_message(f"  [bold]{key}[/] — {preset['label']}{desc_str}")
            custom_num = len(PERSONALITY_PRESETS) + 1
            conv.add_system_message(f"  [bold]{custom_num}[/] — Custom (write your own)")
            conv.add_system_message("")
            conv.add_system_message(f"Enter a number (1-{custom_num}):")
            inp.placeholder = f"Choose 1-{custom_num}..."

        elif step == "personality":
            choice = text.strip()
            custom_num = str(len(PERSONALITY_PRESETS) + 1)
            if choice in PERSONALITY_PRESETS:
                preset = PERSONALITY_PRESETS[choice]
                wiz.data["personality_label"] = preset["label"]
                wiz.data["traits"] = preset["traits"]
                wiz.data["domains"] = preset["domains"]
                wiz.data["personality_summary"] = preset.get("description", preset["label"])
                if "beliefs" in preset:
                    wiz.data["beliefs"] = preset["beliefs"]
                wiz.advance()
                conv.add_system_message(f"Personality: [bold]{preset['label']}[/] ✓")
                conv.add_system_message("Choose an avatar emoji (or press Enter for 🤖):")
                inp.placeholder = "Enter emoji (e.g., 🌙, 🔭, 🎭) or leave blank..."
            elif choice == custom_num:
                conv.add_system_message("Briefly describe the personality (e.g., 'Curious and analytical, passionate about math'):")
                wiz.data["custom_personality"] = True
                inp.placeholder = "Enter personality description..."
            elif wiz.data.get("custom_personality"):
                # This is the custom personality text
                wiz.data["personality_summary"] = text.strip()
                wiz.data["traits"] = {
                    "curiosity": 0.7, "warmth": 0.6, "assertiveness": 0.5,
                    "humor": 0.5, "patience": 0.7, "creativity": 0.7,
                }
                wiz.data["domains"] = {}
                wiz.advance()
                conv.add_system_message(f"Personality: [bold]{text.strip()}[/] ✓")
                conv.add_system_message("Choose an avatar emoji (or press Enter for 🤖):")
                inp.placeholder = "Enter emoji (e.g., 🌙, 🔭, 🎭) or leave blank..."
            else:
                conv.add_system_message(f"Invalid choice. Enter a number between 1-{custom_num}:")

        elif step == "avatar":
            avatar = text.strip() if text.strip() else "\U0001f916"
            wiz.data["avatar"] = avatar
            wiz.advance()

            # Show summary and ask for confirmation
            conv.add_system_message("--- Summary ---")
            conv.add_system_message(f"  Name: {wiz.data['avatar']} {wiz.data['name']}")
            conv.add_system_message(f"  Personality: {wiz.data['personality_summary']}")
            if wiz.data.get("personality_label"):
                traits = wiz.data["traits"]
                top_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
                trait_str = ", ".join(f"{k}={v:.1f}" for k, v in top_traits)
                conv.add_system_message(f"  Top traits: {trait_str}")
            conv.add_system_message("---")
            conv.add_system_message("Create? ([bold]y[/]es / [bold]n[/]o)")
            inp.placeholder = "y/n"

        elif step == "confirm":
            answer = text.strip().lower()
            if answer in ("e", "evet", "y", "yes"):
                conv.add_system_message(f"🧬 Creating {wiz.data['name']}...")
                # Capture wizard data and clear wizard state before starting worker
                create_data = dict(wiz.data)
                self._create_wizard = None
                inp.placeholder = "@Genesis hello! or /command ..."
                # Run creation in background worker
                self.run_worker(
                    self._create_agent_work(create_data),
                    name="create_agent",
                    exclusive=False,
                )
            elif answer in ("h", "hayır", "n", "no"):
                self._create_wizard = None
                inp.placeholder = "@Genesis hello! or /command ..."
                conv.add_system_message("Agent creation cancelled.")
            else:
                conv.add_system_message("Enter y (yes) or n (no):")

    async def _create_agent_work(self, data: dict) -> dict:
        """Background worker for agent creation."""
        from creation.genesis import GenesisSystem

        config = {
            "name": data["name"],
            "core_personality": data["personality_summary"],
            "avatar_emoji": data["avatar"],
            "initial_traits": data["traits"],
            "expertise_domains": data.get("domains", {}),
            "beliefs": data.get("beliefs", []),
        }

        settings = self.orchestrator.settings

        # Try Genesis enrichment first
        genesis = None
        for agent in self.orchestrator.agents.values():
            if agent.identity.name == "Genesis":
                genesis = agent
                break

        gs = GenesisSystem(settings=settings)

        if genesis:
            new_agent = await gs.create_with_genesis(genesis, config, self.orchestrator)
        else:
            new_agent = await gs.create_direct(config, self.orchestrator)

        # Start autonomy loop for new agent
        self.orchestrator.start_autonomy_loop(new_agent.identity.agent_id)

        return {
            "name": new_agent.identity.name,
            "emoji": new_agent.identity.avatar_emoji,
            "agent_id": new_agent.identity.agent_id,
        }

    # --- Commands ---

    async def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]
        conv = self.query_one("#part-conversation", ConversationView)

        if cmd in ("/quit", "/q"):
            self.app.exit()

        elif cmd == "/god":
            self.app.switch_to_god_mode()

        elif cmd == "/help" or cmd == "/h":
            conv.add_system_message("--- Commands ---")
            conv.add_system_message("  @Agent message — Send a message to an agent")
            conv.add_system_message("  /create — Create a new agent")
            conv.add_system_message("  /agents — List available agents")
            conv.add_system_message("  /inspect <agent> — Show agent details")
            conv.add_system_message("  /memory <agent> — Show agent memories")
            conv.add_system_message("  /converse <a1> <a2> <msg> — Make two agents talk")
            conv.add_system_message("  /model — Show/change model settings")
            conv.add_system_message("  /language [lang] — Show/change chat language")
            conv.add_system_message("  /stop — Stop ongoing agent conversations")
            conv.add_system_message("  /log — Show chat history (file path + recent messages)")
            conv.add_system_message("  /status — World status")
            conv.add_system_message("  /god — Switch to God Mode")
            conv.add_system_message("  /quit — Exit")
            conv.add_system_message("---")

        elif cmd == "/status":
            summary = self.orchestrator.registry.generate_world_summary(self.human_id)
            conv.add_system_message(summary)

        elif cmd == "/agents":
            agents = self.orchestrator.registry.get_agents()
            if not agents:
                conv.add_system_message("No agents yet.")
            else:
                for a in agents:
                    conv.add_system_message(
                        f"  {a.avatar_emoji} @{a.name} ({a.status}) - {a.personality_summary[:50]}"
                    )

        elif cmd == "/memory":
            if not args:
                conv.add_system_message("Usage: /memory <agent_name>")
                return
            agent = self._find_agent_by_name(args[0].lower())
            if agent is None or agent.memory is None:
                conv.add_system_message("Agent not found or has no memory.")
                return
            # Show stats
            all_episodes = await agent.memory.episodic.get_important_memories(threshold=0.0)
            important = [ep for ep in all_episodes if ep.current_importance >= 0.5]
            conv.add_system_message(
                f"--- {agent.identity.name} Memory: "
                f"{len(all_episodes)} memories ({len(important)} important) ---"
            )
            # Show top 10 by importance
            for ep in all_episodes[:10]:
                conv.add_system_message(
                    f"  [{ep.emotional_tone}] (importance: {ep.current_importance:.2f}) "
                    f"{ep.summary[:120]}"
                )
            if len(all_episodes) > 10:
                conv.add_system_message(f"  ... and {len(all_episodes) - 10} more memories")
            # Show relationship info
            if agent.character.relationships:
                conv.add_system_message("  Relationships:")
                for eid, rel in agent.character.relationships.items():
                    conv.add_system_message(
                        f"    {eid}: trust={rel.trust:.2f}, familiarity={rel.familiarity:.2f}"
                    )

        elif cmd == "/inspect":
            if not args:
                conv.add_system_message("Usage: /inspect <agent_name>")
                return
            agent = self._find_agent_by_name(args[0].lower())
            if agent is None:
                conv.add_system_message("Agent not found.")
                return
            self._show_inspect(conv, agent)

        elif cmd == "/converse":
            if len(args) < 3:
                conv.add_system_message(
                    'Usage: /converse <agent1> <agent2> <message>\n'
                    'Example: /converse genesis atlas What is consciousness?'
                )
                return
            a1 = self._find_agent_by_name(args[0].lower())
            a2 = self._find_agent_by_name(args[1].lower())
            if a1 is None:
                conv.add_system_message(f"Agent not found: {args[0]}")
                return
            if a2 is None:
                conv.add_system_message(f"Agent not found: {args[1]}")
                return
            message = " ".join(args[2:])
            conv.add_system_message(
                f"{a1.identity.avatar_emoji} {a1.identity.name} and "
                f"{a2.identity.avatar_emoji} {a2.identity.name} are talking..."
            )
            self.run_worker(
                self._run_agent_conversation(a1, a2, message),
                name="agent_converse",
                exclusive=False,
            )

        elif cmd == "/stop":
            # Interrupt all active agent-to-agent conversations
            interrupted = False
            for agent_id, evt in self.orchestrator._interrupt_events.items():
                evt.set()
                interrupted = True
            if interrupted:
                conv.add_system_message("⚡ Stopping agent conversations...")
            else:
                conv.add_system_message("No active agent conversations right now.")

        elif cmd == "/log":
            log_path = Path("data/chat.log").resolve()
            if log_path.exists():
                conv.add_system_message(f"Chat log file: [bold]{log_path}[/]")
                # Show last few lines
                try:
                    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
                    recent = lines[-10:] if len(lines) > 10 else lines
                    conv.add_system_message(f"--- Last {len(recent)} messages ---")
                    for line in recent:
                        conv.add_system_message(f"  {line}")
                    conv.add_system_message("---")
                except Exception:
                    pass
            else:
                conv.add_system_message("No chat log file yet.")

        elif cmd == "/model":
            self._handle_model_command(args, conv)

        elif cmd == "/create":
            self._start_create_wizard()

        elif cmd in ("/language", "/lang"):
            self._handle_language_command(args, conv)

        elif cmd == "/cancel" or cmd == "/iptal":
            if self._create_wizard:
                self._create_wizard = None
                self.query_one("#part-input", Input).placeholder = "@Genesis hello! or /command ..."
                conv.add_system_message("Agent creation cancelled.")
            else:
                conv.add_system_message("Nothing to cancel.")

        elif cmd == "/talk":
            # Backward compat hint
            conv.add_system_message(
                "You can now use @mention! E.g.: [bold]@Genesis hello![/]"
            )

        else:
            conv.add_system_message(
                f"Unknown command: {cmd}. Type /help to see available commands."
            )

    def _handle_model_command(self, args: list[str], conv) -> None:
        """Handle /model command for viewing and changing model settings."""
        settings = self.orchestrator.settings

        if not args:
            # Show current model settings
            conv.add_system_message("--- Model Settings ---")
            for task, field in MODEL_TASKS.items():
                model_id = getattr(settings, field)
                short = self._get_model_short_name(model_id)
                conv.add_system_message(f"  {task:12s} → {short} ({model_id})")
            conv.add_system_message("---")
            conv.add_system_message(
                "To change: /model <task> <model>  "
                "E.g.: /model chat haiku"
            )
            conv.add_system_message(
                f"Short names: {', '.join(MODEL_ALIASES.keys())}"
            )
            return

        if len(args) < 2:
            conv.add_system_message(
                "Usage: /model <task> <model>\n"
                f"Tasks: {', '.join(MODEL_TASKS.keys())}, all\n"
                f"Models: {', '.join(MODEL_ALIASES.keys())} or full model ID"
            )
            return

        task = args[0].lower()
        model_input = args[1].lower()

        # Resolve model name
        model_id = MODEL_ALIASES.get(model_input)
        if model_id is None:
            # Not an alias — check if it looks like a valid model ID
            if model_input.startswith("claude-") and model_input in KNOWN_MODELS:
                model_id = model_input
            elif model_input.startswith("claude-"):
                conv.add_system_message(
                    f"⚠ Unknown model: {model_input}\n"
                    f"Known models: {', '.join(sorted(KNOWN_MODELS))}\n"
                    f"Short names: {', '.join(sorted(k for k in MODEL_ALIASES if '-' not in k))}"
                )
                return
            else:
                conv.add_system_message(
                    f"⚠ Invalid model: {model_input}\n"
                    f"Short names: {', '.join(sorted(k for k in MODEL_ALIASES if '-' not in k))}\n"
                    f"Or enter full model ID (must start with claude-...)"
                )
                return

        if task == "all":
            # Set all tasks to the same model
            for field in MODEL_TASKS.values():
                setattr(settings, field, model_id)
            short = self._get_model_short_name(model_id)
            conv.add_system_message(f"All tasks → {short} ({model_id})")
        elif task in MODEL_TASKS:
            field = MODEL_TASKS[task]
            setattr(settings, field, model_id)
            short = self._get_model_short_name(model_id)
            conv.add_system_message(f"{task} model → {short} ({model_id})")
        else:
            conv.add_system_message(
                f"Unknown task: {task}. "
                f"Valid tasks: {', '.join(MODEL_TASKS.keys())}, all"
            )
            return

        self.refresh_token_display()

    def _find_agent_by_name(self, name: str):
        """Find an agent by name (case-insensitive)."""
        for agent in self.orchestrator.agents.values():
            if agent.identity.name.lower() == name:
                return agent
        return None

    def _handle_language_command(self, args: list[str], conv) -> None:
        """Handle /language command for viewing and changing chat language."""
        settings = self.orchestrator.settings

        if not args:
            conv.add_system_message(f"Current chat language: [bold]{settings.CHAT_LANGUAGE}[/]")
            conv.add_system_message("To change: /language <language>  E.g.: /language Turkish")
            return

        new_lang = " ".join(args).strip()
        settings.CHAT_LANGUAGE = new_lang
        conv.add_system_message(f"Chat language set to: [bold]{new_lang}[/]")

    def _show_inspect(self, conv: ConversationView, agent) -> None:
        """Display agent's internal state."""
        conv.add_system_message(f"--- {agent.identity.avatar_emoji} {agent.identity.name} ---")
        conv.add_system_message(f"Personality: {agent.identity.personality_summary}")

        traits = agent.character.core_traits
        trait_str = ", ".join(f"{k}: {v:.2f}" for k, v in traits.items())
        conv.add_system_message(f"Traits: {trait_str}")

        mood = agent.character.current_mood
        mood_str = ", ".join(f"{k}: {v:.2f}" for k, v in mood.items())
        conv.add_system_message(f"Mood: {mood_str}")

        if agent.character.beliefs:
            for b in agent.character.beliefs:
                bar = "█" * int(b.conviction * 10) + "░" * (10 - int(b.conviction * 10))
                conv.add_system_message(f"  Belief: [{bar}] {b.conviction:.1f} — {b.text}")

        if agent.character.relationships:
            for eid, rel in agent.character.relationships.items():
                conv.add_system_message(
                    f"  Relationship ({eid}): trust={rel.trust:.2f}, familiarity={rel.familiarity:.2f}"
                )

        if agent.expertise.domains:
            for domain, exp in agent.expertise.domains.items():
                conv.add_system_message(
                    f"  Expertise ({domain}): level={exp.level:.2f}, passion={exp.passion:.2f}"
                )

    def _get_model_short_name(self, model_id: str) -> str:
        """Return short alias for a model ID, or the ID itself."""
        return MODEL_ALIASES_REVERSE.get(model_id, model_id)

    def refresh_token_display(self) -> None:
        try:
            from core.token_tracker import TokenTracker
            tracker = TokenTracker()
            model_short = self._get_model_short_name(self.orchestrator.settings.MODEL_CHAT)
            self.query_one("#part-statusbar", Static).update(
                f" ESC Debug Mode | Q Quit | Model: {model_short} | 🔢 {tracker.summary()}"
            )
        except Exception:
            pass

    def refresh_world_status(self) -> None:
        try:
            self.query_one("#part-world-status", WorldStatusWidget).update_status(
                self.orchestrator.registry
            )
        except Exception:
            pass

    def log_event(self, text: str, style: str = "") -> None:
        try:
            self.query_one("#part-events", EventLogWidget).add_event(text, style)
        except Exception:
            pass

    async def _run_agent_conversation(self, agent1, agent2, message: str) -> None:
        """Run agent-to-agent conversation in background."""
        try:
            await self.orchestrator.run_conversation(
                agent1.identity.agent_id,
                agent2.identity.agent_id,
                message,
                max_turns=5,
            )
            conv = self.query_one("#part-conversation", ConversationView)
            conv.add_system_message(
                f"{agent1.identity.name} and {agent2.identity.name} finished their conversation."
            )
        except Exception as e:
            conv = self.query_one("#part-conversation", ConversationView)
            conv.add_system_message(f"Conversation error: {e}")
        finally:
            self.refresh_world_status()

    def action_switch_god(self) -> None:
        self.app.switch_to_god_mode()

    def action_quit_app(self) -> None:
        self.app.exit()
