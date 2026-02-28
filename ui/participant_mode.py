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
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker, WorkerState

from ui.widgets import ConversationView, EventLogWidget, WorldStatusWidget

if TYPE_CHECKING:
    from world.orchestrator import Orchestrator


# Preset personality templates for quick agent creation
PERSONALITY_PRESETS = {
    "1": {
        "label": "Analitik Düşünür",
        "traits": {"curiosity": 0.9, "warmth": 0.4, "assertiveness": 0.7, "humor": 0.3, "patience": 0.8, "creativity": 0.6},
        "domains": {"mathematics": {"level": 0.8, "passion": 0.9, "style": "analytical"}, "logic": {"level": 0.7, "passion": 0.8, "style": "analytical"}},
    },
    "2": {
        "label": "Empatik Dinleyici",
        "traits": {"curiosity": 0.7, "warmth": 0.95, "assertiveness": 0.3, "humor": 0.5, "patience": 0.95, "creativity": 0.6},
        "domains": {"psychology": {"level": 0.8, "passion": 0.9, "style": "empathetic"}, "communication": {"level": 0.7, "passion": 0.8, "style": "empathetic"}},
    },
    "3": {
        "label": "Yaratıcı Ruh",
        "traits": {"curiosity": 0.8, "warmth": 0.7, "assertiveness": 0.5, "humor": 0.8, "patience": 0.6, "creativity": 0.95},
        "domains": {"art": {"level": 0.8, "passion": 0.95, "style": "intuitive"}, "creativity": {"level": 0.9, "passion": 0.9, "style": "creative"}},
    },
    "4": {
        "label": "Bilge Filozof",
        "traits": {"curiosity": 0.95, "warmth": 0.6, "assertiveness": 0.4, "humor": 0.5, "patience": 0.9, "creativity": 0.8},
        "domains": {"philosophy": {"level": 0.9, "passion": 0.95, "style": "socratic"}, "ethics": {"level": 0.7, "passion": 0.8, "style": "socratic"}},
    },
    "5": {
        "label": "Hoppa Enerji Bombası",
        "description": "Konudan konuya atlayan, sabırsız ama bulaşıcı enerjili tip",
        "traits": {"curiosity": 0.95, "warmth": 0.8, "assertiveness": 0.7, "humor": 0.9, "patience": 0.1, "creativity": 0.9},
        "domains": {"entertainment": {"level": 0.7, "passion": 0.9, "style": "intuitive"}, "gossip": {"level": 0.8, "passion": 0.8, "style": "intuitive"}},
        "beliefs": ["Hayat çok kısa, sıkılmak günah", "Her an bir macera fırsatı", "Ciddiyet yaratıcılığı öldürür"],
    },
    "6": {
        "label": "Her Şeyi Bilen Cahil",
        "description": "Her konuda fikri var ama bilgisi yok, yanlışı özgüvenle savunur",
        "traits": {"curiosity": 0.15, "warmth": 0.5, "assertiveness": 0.95, "humor": 0.4, "patience": 0.3, "creativity": 0.3},
        "domains": {"history": {"level": 0.2, "passion": 0.9, "style": "analytical"}, "science": {"level": 0.1, "passion": 0.8, "style": "analytical"}},
        "beliefs": ["Ben okumasam da bilirim", "Kitaplar gerçeği söylemez, hayat söyler", "Herkes yanlış biliyor, ben doğrusunu biliyorum"],
    },
    "7": {
        "label": "Sinirli Ama Haklı",
        "description": "Her şeye sinirli, eleştirileri sert ama isabetli, içi yumuşak dışı sert",
        "traits": {"curiosity": 0.7, "warmth": 0.25, "assertiveness": 0.95, "humor": 0.6, "patience": 0.1, "creativity": 0.5},
        "domains": {"politics": {"level": 0.7, "passion": 0.9, "style": "analytical"}, "ethics": {"level": 0.6, "passion": 0.8, "style": "socratic"}},
        "beliefs": ["Dünya adil değil ve biri bunu söylemeli", "Saygı kazanılır, verilmez", "Kibarlık çoğu zaman ikiyüzlülüktür"],
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
                yield Static("\U0001f30d Dünya", id="part-world-title")
                yield WorldStatusWidget(id="part-world-status")
                yield Static("\U0001f4dc Olaylar", id="part-events-title")
                yield EventLogWidget(id="part-events", wrap=True, markup=True)
            with Vertical(id="part-right"):
                yield Static(
                    "\U0001f4ac Grup Sohbet",
                    id="part-conv-title",
                )
                yield ConversationView(id="part-conversation", wrap=True, markup=True)
                yield Static(
                    "@Agent mesaj | /create | /stop | /agents | /converse | /help | /god",
                    id="part-help",
                )
                yield Input(placeholder="@Genesis merhaba! veya /komut ...", id="part-input")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_world_status()
        self.set_interval(5, self.refresh_world_status)

        conv = self.query_one("#part-conversation", ConversationView)
        conv.add_system_message("Grup sohbete hoş geldiniz!")
        conv.add_system_message(
            "Bir agent'a yazmak için @isim kullanın. Örn: [bold]@Genesis merhaba![/]"
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
            conv.add_system_message(f"Mevcut agent'lar: {', '.join(names)}")

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
                conv.add_system_message("Henüz hiç agent yok.")
                return
            if not text.strip():
                conv = self.query_one("#part-conversation", ConversationView)
                conv.add_system_message("Mesaj boş olamaz.")
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
                    f"⚡ {agent_name} konuşmasından çekildi, size dönüyor..."
                )

        conv.add_system_message(f"{agent_name} düşünüyor...")

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
        conv.add_system_message(f"Herkes düşünüyor... ({agent_names})")

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
                conv.add_system_message(f"Hata: {event.worker.error}")
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
                    f"✅ {result['emoji']} [bold]{result['name']}[/] başarıyla yaratıldı!"
                )
                conv.add_system_message(
                    f"Hemen konuşmaya başlayabilirsiniz: [bold]@{result['name']} merhaba![/]"
                )
                self._show_available_agents(conv)
                self.refresh_world_status()
            elif event.state == WorkerState.ERROR:
                conv.add_system_message(f"❌ Agent yaratma hatası: {event.worker.error}")

    # --- Create Wizard ---

    def _start_create_wizard(self) -> None:
        """Start the interactive agent creation wizard."""
        self._create_wizard = _CreateWizard()
        conv = self.query_one("#part-conversation", ConversationView)
        inp = self.query_one("#part-input", Input)

        conv.add_system_message("--- 🧬 Yeni Agent Yaratma ---")
        conv.add_system_message("İptal etmek için /cancel yazabilirsiniz.\n")
        conv.add_system_message("Agent'ın adı ne olsun?")
        inp.placeholder = "Agent adı girin..."

    async def _handle_create_input(self, text: str) -> None:
        """Handle user input during creation wizard."""
        conv = self.query_one("#part-conversation", ConversationView)
        inp = self.query_one("#part-input", Input)

        # Allow cancel at any step
        if text.strip().lower() in ("/cancel", "/iptal"):
            self._create_wizard = None
            inp.placeholder = "@Genesis merhaba! veya /komut ..."
            conv.add_system_message("Agent yaratma iptal edildi.")
            return

        wiz = self._create_wizard
        step = wiz.current_step

        # Echo user input
        conv.add_user_message("Sen", text)

        if step == "name":
            # Validate name
            name = text.strip()
            if not name:
                conv.add_system_message("İsim boş olamaz. Tekrar deneyin:")
                return
            # Check if name already exists
            if self._find_agent_by_name(name.lower()):
                conv.add_system_message(f"'{name}' adında bir agent zaten var. Başka bir isim girin:")
                return
            wiz.data["name"] = name
            wiz.advance()

            # Show personality presets
            conv.add_system_message(f"Harika! {name} için bir kişilik seçin:")
            conv.add_system_message("")
            for key, preset in PERSONALITY_PRESETS.items():
                desc = preset.get("description", "")
                desc_str = f" — [dim]{desc}[/]" if desc else ""
                conv.add_system_message(f"  [bold]{key}[/] — {preset['label']}{desc_str}")
            custom_num = len(PERSONALITY_PRESETS) + 1
            conv.add_system_message(f"  [bold]{custom_num}[/] — Özel (kendi tanımınızı yazın)")
            conv.add_system_message("")
            conv.add_system_message(f"Numara girin (1-{custom_num}):")
            inp.placeholder = f"1-{custom_num} arası seçin..."

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
                conv.add_system_message(f"Kişilik: [bold]{preset['label']}[/] ✓")
                conv.add_system_message("Avatar emoji seçin (veya Enter'a basarak 🤖 kullanın):")
                inp.placeholder = "Emoji girin (örn: 🌙, 🔭, 🎭) veya boş bırakın..."
            elif choice == custom_num:
                conv.add_system_message("Kişiliği kısaca tanımlayın (örn: 'Meraklı ve analitik, matematik tutkunu'):")
                wiz.data["custom_personality"] = True
                inp.placeholder = "Kişilik tanımı girin..."
            elif wiz.data.get("custom_personality"):
                # This is the custom personality text
                wiz.data["personality_summary"] = text.strip()
                wiz.data["traits"] = {
                    "curiosity": 0.7, "warmth": 0.6, "assertiveness": 0.5,
                    "humor": 0.5, "patience": 0.7, "creativity": 0.7,
                }
                wiz.data["domains"] = {}
                wiz.advance()
                conv.add_system_message(f"Kişilik: [bold]{text.strip()}[/] ✓")
                conv.add_system_message("Avatar emoji seçin (veya Enter'a basarak 🤖 kullanın):")
                inp.placeholder = "Emoji girin (örn: 🌙, 🔭, 🎭) veya boş bırakın..."
            else:
                conv.add_system_message(f"Geçersiz seçim. 1-{custom_num} arası bir numara girin:")

        elif step == "avatar":
            avatar = text.strip() if text.strip() else "\U0001f916"
            wiz.data["avatar"] = avatar
            wiz.advance()

            # Show summary and ask for confirmation
            conv.add_system_message("--- Özet ---")
            conv.add_system_message(f"  İsim: {wiz.data['avatar']} {wiz.data['name']}")
            conv.add_system_message(f"  Kişilik: {wiz.data['personality_summary']}")
            if wiz.data.get("personality_label"):
                traits = wiz.data["traits"]
                top_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)[:3]
                trait_str = ", ".join(f"{k}={v:.1f}" for k, v in top_traits)
                conv.add_system_message(f"  Öne çıkan: {trait_str}")
            conv.add_system_message("---")
            conv.add_system_message("Yaratılsın mı? ([bold]e[/]vet / [bold]h[/]ayır)")
            inp.placeholder = "e/h"

        elif step == "confirm":
            answer = text.strip().lower()
            if answer in ("e", "evet", "y", "yes"):
                conv.add_system_message(f"🧬 {wiz.data['name']} yaratılıyor...")
                # Capture wizard data and clear wizard state before starting worker
                create_data = dict(wiz.data)
                self._create_wizard = None
                inp.placeholder = "@Genesis merhaba! veya /komut ..."
                # Run creation in background worker
                self.run_worker(
                    self._create_agent_work(create_data),
                    name="create_agent",
                    exclusive=False,
                )
            elif answer in ("h", "hayır", "n", "no"):
                self._create_wizard = None
                inp.placeholder = "@Genesis merhaba! veya /komut ..."
                conv.add_system_message("Agent yaratma iptal edildi.")
            else:
                conv.add_system_message("e (evet) veya h (hayır) girin:")

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
            conv.add_system_message("--- Komutlar ---")
            conv.add_system_message("  @Agent mesaj — Bir agent'a mesaj gönder")
            conv.add_system_message("  /create — Yeni agent yarat")
            conv.add_system_message("  /agents — Mevcut agent'ları listele")
            conv.add_system_message("  /inspect <agent> — Agent detaylarını göster")
            conv.add_system_message("  /memory <agent> — Agent anılarını göster")
            conv.add_system_message("  /converse <a1> <a2> <mesaj> — İki agent'ı konuştur")
            conv.add_system_message("  /stop — Devam eden agent konuşmalarını durdur")
            conv.add_system_message("  /log — Chat geçmişini göster (dosya yolu + son mesajlar)")
            conv.add_system_message("  /status — Dünya durumu")
            conv.add_system_message("  /god — God Mode'a geç")
            conv.add_system_message("  /quit — Çıkış")
            conv.add_system_message("---")

        elif cmd == "/status":
            summary = self.orchestrator.registry.generate_world_summary(self.human_id)
            conv.add_system_message(summary)

        elif cmd == "/agents":
            agents = self.orchestrator.registry.get_agents()
            if not agents:
                conv.add_system_message("Henüz agent yok.")
            else:
                for a in agents:
                    conv.add_system_message(
                        f"  {a.avatar_emoji} @{a.name} ({a.status}) - {a.personality_summary[:50]}"
                    )

        elif cmd == "/memory":
            if not args:
                conv.add_system_message("Kullanım: /memory <agent_adı>")
                return
            agent = self._find_agent_by_name(args[0].lower())
            if agent is None or agent.memory is None:
                conv.add_system_message("Agent bulunamadı veya hafızası yok.")
                return
            # Show stats
            all_episodes = await agent.memory.episodic.get_important_memories(threshold=0.0)
            important = [ep for ep in all_episodes if ep.current_importance >= 0.5]
            conv.add_system_message(
                f"--- {agent.identity.name} Hafızası: "
                f"{len(all_episodes)} anı ({len(important)} önemli) ---"
            )
            # Show top 10 by importance
            for ep in all_episodes[:10]:
                conv.add_system_message(
                    f"  [{ep.emotional_tone}] (önem: {ep.current_importance:.2f}) "
                    f"{ep.summary[:120]}"
                )
            if len(all_episodes) > 10:
                conv.add_system_message(f"  ... ve {len(all_episodes) - 10} anı daha")
            # Show relationship info
            if agent.character.relationships:
                conv.add_system_message("  İlişkiler:")
                for eid, rel in agent.character.relationships.items():
                    conv.add_system_message(
                        f"    {eid}: güven={rel.trust:.2f}, aşinalık={rel.familiarity:.2f}"
                    )

        elif cmd == "/inspect":
            if not args:
                conv.add_system_message("Kullanım: /inspect <agent_adı>")
                return
            agent = self._find_agent_by_name(args[0].lower())
            if agent is None:
                conv.add_system_message("Agent bulunamadı.")
                return
            self._show_inspect(conv, agent)

        elif cmd == "/converse":
            if len(args) < 3:
                conv.add_system_message(
                    'Kullanım: /converse <agent1> <agent2> <mesaj>\n'
                    'Örnek: /converse genesis atlas Bilinç nedir?'
                )
                return
            a1 = self._find_agent_by_name(args[0].lower())
            a2 = self._find_agent_by_name(args[1].lower())
            if a1 is None:
                conv.add_system_message(f"Agent bulunamadı: {args[0]}")
                return
            if a2 is None:
                conv.add_system_message(f"Agent bulunamadı: {args[1]}")
                return
            message = " ".join(args[2:])
            conv.add_system_message(
                f"{a1.identity.avatar_emoji} {a1.identity.name} ve "
                f"{a2.identity.avatar_emoji} {a2.identity.name} konuşuyor..."
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
                conv.add_system_message("⚡ Agent konuşmaları durduruluyor...")
            else:
                conv.add_system_message("Şu anda aktif bir agent konuşması yok.")

        elif cmd == "/log":
            log_path = Path("data/chat.log").resolve()
            if log_path.exists():
                conv.add_system_message(f"Chat log dosyası: [bold]{log_path}[/]")
                # Show last few lines
                try:
                    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
                    recent = lines[-10:] if len(lines) > 10 else lines
                    conv.add_system_message(f"--- Son {len(recent)} mesaj ---")
                    for line in recent:
                        conv.add_system_message(f"  {line}")
                    conv.add_system_message("---")
                except Exception:
                    pass
            else:
                conv.add_system_message("Henüz chat log dosyası oluşmadı.")

        elif cmd == "/create":
            self._start_create_wizard()

        elif cmd == "/cancel" or cmd == "/iptal":
            if self._create_wizard:
                self._create_wizard = None
                self.query_one("#part-input", Input).placeholder = "@Genesis merhaba! veya /komut ..."
                conv.add_system_message("Agent yaratma iptal edildi.")
            else:
                conv.add_system_message("İptal edilecek bir işlem yok.")

        elif cmd == "/talk":
            # Backward compat hint
            conv.add_system_message(
                "Artık @mention kullanabilirsiniz! Örn: [bold]@Genesis merhaba![/]"
            )

        else:
            conv.add_system_message(
                f"Bilinmeyen komut: {cmd}. /help yazarak komutları görebilirsiniz."
            )

    def _find_agent_by_name(self, name: str):
        """Find an agent by name (case-insensitive)."""
        for agent in self.orchestrator.agents.values():
            if agent.identity.name.lower() == name:
                return agent
        return None

    def _show_inspect(self, conv: ConversationView, agent) -> None:
        """Display agent's internal state."""
        conv.add_system_message(f"--- {agent.identity.avatar_emoji} {agent.identity.name} ---")
        conv.add_system_message(f"Kişilik: {agent.identity.personality_summary}")

        traits = agent.character.core_traits
        trait_str = ", ".join(f"{k}: {v:.2f}" for k, v in traits.items())
        conv.add_system_message(f"Özellikler: {trait_str}")

        mood = agent.character.current_mood
        mood_str = ", ".join(f"{k}: {v:.2f}" for k, v in mood.items())
        conv.add_system_message(f"Ruh hali: {mood_str}")

        if agent.character.beliefs:
            conv.add_system_message(f"İnançlar: {'; '.join(agent.character.beliefs)}")

        if agent.character.relationships:
            for eid, rel in agent.character.relationships.items():
                conv.add_system_message(
                    f"  İlişki ({eid}): güven={rel.trust:.2f}, aşinalık={rel.familiarity:.2f}"
                )

        if agent.expertise.domains:
            for domain, exp in agent.expertise.domains.items():
                conv.add_system_message(
                    f"  Uzmanlık ({domain}): seviye={exp.level:.2f}, tutku={exp.passion:.2f}"
                )

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
                f"{agent1.identity.name} ve {agent2.identity.name} konuşmasını bitirdi."
            )
        except Exception as e:
            conv = self.query_one("#part-conversation", ConversationView)
            conv.add_system_message(f"Konuşma hatası: {e}")
        finally:
            self.refresh_world_status()

    def action_switch_god(self) -> None:
        self.app.switch_to_god_mode()

    def action_quit_app(self) -> None:
        self.app.exit()
