"""Orchestrator — main control class managing the entire Living Agents system.

Coordinates agents, conversations, the message bus, world registry,
shared state, and autonomy loops.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import anthropic

from config.settings import Settings
from conversation.engine import ConversationEngine
from conversation.reflection import ReflectionEngine
from core.agent import Agent
from core.character import CharacterState
from core.expertise import ExpertiseSystem
from core.identity import AgentIdentity
from memory.database import get_db, init_database
from memory.store import MemoryStore
from world.message_bus import Message, MessageBus
from world.registry import WorldEntity, WorldRegistry
from world.shared_state import SharedWorldState, WorldEvent

logger = logging.getLogger(__name__)


class Orchestrator:
    """Top-level coordinator for the Living Agents system."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.registry = WorldRegistry()
        self.message_bus = MessageBus(db_path=self.settings.DB_PATH)
        self.shared_state = SharedWorldState(db_path=self.settings.DB_PATH)
        self.reflection_engine = ReflectionEngine(settings=self.settings)

        self.agents: dict[str, Agent] = {}
        self.conversation_engines: dict[str, ConversationEngine] = {}

        self._autonomy_tasks: dict[str, asyncio.Task] = {}
        self._running = False

        # Interrupt mechanism: maps agent_id → asyncio.Event
        # When set, any run_conversation involving that agent stops after the current turn
        self._interrupt_events: dict[str, asyncio.Event] = {}
        # Stores the human message that caused the interrupt
        self._interrupt_messages: dict[str, tuple[str, str]] = {}  # agent_id → (human_id, message)

        # Callbacks for UI integration
        self._on_event: list = []  # list of async/sync callables: (event_text, event_type)
        self._on_conversation_message: list = []  # (speaker_name, message, emoji)

    async def start(self) -> None:
        """Initialize the system: database, load agents, start autonomy loops."""
        await init_database(self.settings.DB_PATH)

        # Load existing agents from database
        await self._load_agents()

        self._running = True
        logger.info(
            "Orchestrator started (%d agents loaded)",
            len(self.agents),
        )

    async def stop(self) -> None:
        """Graceful shutdown: end conversations, save state, cancel loops."""
        self._running = False

        # Cancel autonomy loops
        for task in self._autonomy_tasks.values():
            task.cancel()
        if self._autonomy_tasks:
            await asyncio.gather(*self._autonomy_tasks.values(), return_exceptions=True)
        self._autonomy_tasks.clear()

        # End all active conversations
        for engine in self.conversation_engines.values():
            try:
                await engine.end_conversation()
            except Exception:
                logger.exception("Error ending conversation")

        # Save all agent states
        await self._save_all_agents()

        logger.info("Orchestrator stopped")

    async def create_agent(
        self,
        config: dict[str, Any],
        created_by: str = "system",
    ) -> Agent:
        """Create a new agent and register it in the world.

        Config should contain: name, personality_summary, avatar_emoji,
        and optionally: core_traits, current_mood, beliefs, domains,
        teaching_style, learning_rate.
        """
        # Build identity
        agent_id = str(uuid4())
        identity = AgentIdentity(
            agent_id=agent_id,
            name=config.get("name", "Isimsiz"),
            created_by=created_by,
            personality_summary=config.get("personality_summary", ""),
            avatar_emoji=config.get("avatar_emoji", "\U0001f916"),
        )

        # Build character state
        character = CharacterState()
        if "core_traits" in config:
            character.core_traits.update(config["core_traits"])
        if "current_mood" in config:
            character.current_mood.update(config["current_mood"])
        for belief in config.get("beliefs", []):
            character.add_belief(belief)

        # Build expertise
        from core.expertise import DomainExpertise
        expertise = ExpertiseSystem(
            learning_rate=config.get("learning_rate", 0.5),
            teaching_style=config.get("teaching_style", "step_by_step"),
        )
        for domain_name, domain_config in config.get("domains", {}).items():
            if isinstance(domain_config, dict):
                expertise.domains[domain_name] = DomainExpertise(**domain_config)

        # Create memory store
        memory = MemoryStore(
            agent_id=agent_id,
            db_path=self.settings.DB_PATH,
            chroma_path=self.settings.CHROMA_PATH,
            max_tokens=self.settings.MAX_CONTEXT_TOKENS,
        )
        await memory.init()

        # Assemble agent
        # Rebuild model to resolve forward ref before instantiation
        Agent.model_rebuild()
        agent = Agent(
            identity=identity,
            character=character,
            expertise=expertise,
            memory=memory,
        )

        # Register in world
        self.agents[agent_id] = agent
        self._register_entity(agent)

        # Create conversation engine
        self._create_engine(agent)

        # Create message bus inbox
        self.message_bus.create_inbox(agent_id)

        # Save to database
        await self._save_agent(agent)

        # Record creation event
        event_text = f"{identity.name} yaratıldı (yaratıcı: {created_by})"
        await self.shared_state.add_event(WorldEvent(
            event=event_text,
            participants=[agent_id, created_by],
            event_type="creation",
        ))
        await self._fire_event(event_text, "creation")

        # Notify others
        notify_ids = self.registry.notify_all(
            f"{identity.name} dünyaya katıldı!", exclude=agent_id,
        )
        for eid in notify_ids:
            await self.message_bus.send(Message(
                from_id="system",
                to_id=eid,
                message_type="notification",
                content=f"{identity.name} dünyaya katıldı!",
            ))

        # Create "first awakening" memory
        from memory.episodic import Episode
        awakening = Episode(
            agent_id=agent_id,
            participants=[agent_id],
            summary=f"Ben {identity.name}, ilk kez uyanıyorum. {identity.personality_summary}",
            emotional_tone="heyecan",
            key_facts=[f"Yaratıcım: {created_by}", f"Adım: {identity.name}"],
            importance=0.9,
            current_importance=0.9,
            tags=["yaratılış", "ilk_anı"],
        )
        await memory.save_episode(awakening)

        logger.info("Agent created: %s (%s)", identity.name, agent_id)
        return agent

    async def handle_human_message(
        self,
        human_id: str,
        target_agent_id: str,
        message: str,
    ) -> str:
        """Handle a message from a human to an agent.

        Messages are always delivered immediately — group chat model.
        If the agent is in an agent-to-agent conversation (run_conversation),
        that conversation is interrupted so the human gets priority.
        """
        agent = self.agents.get(target_agent_id)
        if agent is None:
            raise ValueError(f"Agent {target_agent_id} not found")

        engine = self.conversation_engines.get(target_agent_id)
        if engine is None:
            raise ValueError(f"No conversation engine for agent {target_agent_id}")

        # If agent is in a run_conversation loop, signal interrupt
        if target_agent_id in self._interrupt_events:
            self._interrupt_conversation(target_agent_id, human_id, message)
            await asyncio.sleep(0.3)

        # Get response — always delivered
        response = await engine.chat(message, sender_id=human_id)

        return response

    def _interrupt_conversation(self, agent_id: str, human_id: str, message: str) -> None:
        """Signal an agent-to-agent conversation to stop."""
        # Set interrupt event for the agent
        if agent_id in self._interrupt_events:
            self._interrupt_events[agent_id].set()
        # Also interrupt the partner
        entity = self.registry.get(agent_id)
        if entity and entity.current_conversation_with:
            partner_id = entity.current_conversation_with
            if partner_id in self._interrupt_events:
                self._interrupt_events[partner_id].set()
        logger.info("Interrupting conversation for agent %s (human message from %s)", agent_id, human_id)

    async def handle_agent_to_agent(
        self,
        from_id: str,
        to_id: str,
        message: str,
    ) -> str:
        """Handle a message from one agent to another.

        Messages are always delivered — no availability check.
        Group chat model: agents can receive messages from anyone at any time.
        """
        from_agent = self.agents.get(from_id)
        to_agent = self.agents.get(to_id)
        if from_agent is None or to_agent is None:
            raise ValueError(f"Agent not found: from={from_id}, to={to_id}")

        to_engine = self.conversation_engines.get(to_id)
        if to_engine is None:
            raise ValueError(f"No conversation engine for agent {to_id}")

        # Get response — no blocking, messages always go through
        response = await to_engine.chat(message, sender_id=from_id)

        return response

    # Signal that an agent appends to their message when the conversation
    # has reached a natural conclusion.
    CONVERSATION_END_SIGNAL = "[VEDA]"

    async def run_conversation(
        self,
        agent1_id: str,
        agent2_id: str,
        initiator_message: str,
        max_turns: int = 6,
    ) -> list[dict[str, str]]:
        """Run a full agent-to-agent conversation.

        Agents decide when the conversation naturally ends by appending
        [VEDA] to their message. max_turns is a hard safety limit (default 6).

        Returns the conversation transcript as a list of
        {"speaker": agent_name, "message": text} dicts.
        """
        agent1 = self.agents.get(agent1_id)
        agent2 = self.agents.get(agent2_id)
        if agent1 is None or agent2 is None:
            raise ValueError("One or both agents not found")

        engine1 = self.conversation_engines.get(agent1_id)
        engine2 = self.conversation_engines.get(agent2_id)
        if engine1 is None or engine2 is None:
            raise ValueError("Missing conversation engines")

        # Reset engines for fresh conversation
        engine1.reset()
        engine2.reset()

        # Inject conversation rules into working memory context
        end_instruction = (
            f"[Sistem notu: {agent1.identity.name} ile {agent2.identity.name} arasında "
            f"KISA bir sohbet başlıyor. KURALLAR:\n"
            f"1. KISA KONUŞ — her mesajın EN FAZLA 1-2 cümle olsun.\n"
            f"2. Bu sohbet en fazla {max_turns} tur sürecek. Uzatma, hızlıca konuyu kapat.\n"
            f"3. Söyleyeceğini söyle, vedalaş, mesajının sonuna {self.CONVERSATION_END_SIGNAL} ekle.\n"
            f"4. Her turda kenine sor: 'Söyleyecek yeni bir şey var mı?' Yoksa HEMEN bitir.]"
        )
        engine1.agent.memory.working.add_message("user", end_instruction)
        engine1.agent.memory.working.add_message("assistant", "Anladım, kısa konuşacağım.")
        engine2.agent.memory.working.add_message("user", end_instruction)
        engine2.agent.memory.working.add_message("assistant", "Anladım, kısa konuşacağım.")

        # Setup interrupt events for both agents
        interrupt1 = asyncio.Event()
        interrupt2 = asyncio.Event()
        self._interrupt_events[agent1_id] = interrupt1
        self._interrupt_events[agent2_id] = interrupt2

        # Update statuses
        self.registry.update_status(agent1_id, "in_conversation", conversation_with=agent2_id)
        self.registry.update_status(agent2_id, "in_conversation", conversation_with=agent1_id)

        transcript: list[dict[str, str]] = []
        current_message = initiator_message
        actual_turns = 0
        interrupted = False

        # Fire the initiator message only once (agent1 starts the conversation)
        await self._fire_conversation(
            agent1.identity.name, initiator_message, agent1.identity.avatar_emoji
        )

        for turn in range(max_turns):
            # Check for human interrupt before each turn
            if interrupt1.is_set() or interrupt2.is_set():
                interrupted = True
                logger.info(
                    "Conversation interrupted by human (%s <-> %s) at turn %d",
                    agent1.identity.name, agent2.identity.name, actual_turns,
                )
                break

            finished = False

            # Append turn reminder to the message so agents don't forget to end
            remaining = max_turns - turn
            turn_msg = current_message
            if remaining <= 3:
                turn_msg = (
                    f"{current_message}\n\n"
                    f"[Sistem: Kalan tur: {remaining}. Konuşmayı bitirme zamanı. "
                    f"Son mesajının sonuna {self.CONVERSATION_END_SIGNAL} ekle.]"
                )
            elif remaining <= max_turns // 2:
                turn_msg = (
                    f"{current_message}\n\n"
                    f"[Sistem: Tur {turn + 1}/{max_turns}. Kısa cevap ver. "
                    f"Konu bittiyse {self.CONVERSATION_END_SIGNAL} ekle.]"
                )

            if turn % 2 == 0:
                # Agent2 responds to agent1's message
                response = await engine2.chat(turn_msg, sender_id=agent1_id)

                clean_response = response
                if self.CONVERSATION_END_SIGNAL in response:
                    clean_response = response.replace(self.CONVERSATION_END_SIGNAL, "").strip()
                    finished = True

                await self._fire_conversation(
                    agent2.identity.name, clean_response, agent2.identity.avatar_emoji
                )
                transcript.append({
                    "speaker": agent1.identity.name,
                    "message": current_message,
                })
                transcript.append({
                    "speaker": agent2.identity.name,
                    "message": clean_response,
                })
            else:
                # Agent1 responds to agent2's previous response
                response = await engine1.chat(turn_msg, sender_id=agent2_id)

                clean_response = response
                if self.CONVERSATION_END_SIGNAL in response:
                    clean_response = response.replace(self.CONVERSATION_END_SIGNAL, "").strip()
                    finished = True

                await self._fire_conversation(
                    agent1.identity.name, clean_response, agent1.identity.avatar_emoji
                )
                transcript.append({
                    "speaker": agent2.identity.name,
                    "message": current_message,
                })
                transcript.append({
                    "speaker": agent1.identity.name,
                    "message": clean_response,
                })

            actual_turns += 1
            current_message = clean_response

            if finished:
                logger.info(
                    "Conversation ended naturally after %d turns (%s <-> %s)",
                    actual_turns, agent1.identity.name, agent2.identity.name,
                )
                break

        # Cleanup interrupt events
        self._interrupt_events.pop(agent1_id, None)
        self._interrupt_events.pop(agent2_id, None)

        # End conversations and trigger reflections
        await engine1.end_conversation()
        await engine2.end_conversation()

        # Reset statuses
        self.registry.update_status(agent1_id, "idle")
        self.registry.update_status(agent2_id, "idle")

        # Record event
        if interrupted:
            end_reason = "insan müdahalesi"
        elif actual_turns < max_turns:
            end_reason = "doğal bitiş"
        else:
            end_reason = f"limit ({max_turns})"
        event_text = (
            f"{agent1.identity.name} ve {agent2.identity.name} konuştu "
            f"({actual_turns} tur, {end_reason})"
        )
        await self.shared_state.add_event(WorldEvent(
            event=event_text,
            participants=[agent1_id, agent2_id],
            event_type="conversation",
        ))
        await self._fire_event(event_text, "conversation")

        logger.info(
            "Agent conversation completed: %s <-> %s (%d turns, %s)",
            agent1.identity.name, agent2.identity.name, actual_turns, end_reason,
        )
        return transcript

    async def autonomy_loop(self, agent_id: str) -> None:
        """Autonomous decision loop for an agent.

        Every AUTONOMY_INTERVAL seconds, asks Claude what the agent wants to do:
        talk_to:<id>, reflect, idle, or create.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return

        client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)

        while self._running:
            await asyncio.sleep(self.settings.AUTONOMY_INTERVAL)

            if not self._running:
                break

            # Skip if agent is busy
            entity = self.registry.get(agent_id)
            if entity and entity.status == "in_conversation":
                continue

            self.registry.update_status(agent_id, "thinking")

            try:
                decision = await self._make_autonomy_decision(agent, client)
                await self._execute_autonomy_decision(agent_id, decision)
            except Exception:
                logger.exception("Autonomy loop error for %s", agent.identity.name)
            finally:
                # Reset to idle if not in conversation
                entity = self.registry.get(agent_id)
                if entity and entity.status == "thinking":
                    self.registry.update_status(agent_id, "idle")

    def start_autonomy_loop(self, agent_id: str) -> None:
        """Start the autonomy loop as a background task."""
        if agent_id in self._autonomy_tasks:
            return
        task = asyncio.create_task(self.autonomy_loop(agent_id))
        self._autonomy_tasks[agent_id] = task

    def stop_autonomy_loop(self, agent_id: str) -> None:
        """Stop an agent's autonomy loop."""
        task = self._autonomy_tasks.pop(agent_id, None)
        if task is not None:
            task.cancel()

    def on_event(self, callback) -> None:
        """Register a callback for world events: callback(event_text, event_type)."""
        self._on_event.append(callback)

    def on_conversation_message(self, callback) -> None:
        """Register a callback for conversation messages: callback(speaker, message, emoji)."""
        self._on_conversation_message.append(callback)

    async def _fire_event(self, text: str, event_type: str = "") -> None:
        """Notify all registered event callbacks."""
        for cb in self._on_event:
            try:
                result = cb(text, event_type)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.debug("Event callback error", exc_info=True)

    async def _fire_conversation(self, speaker: str, message: str, emoji: str = "") -> None:
        """Notify all registered conversation callbacks."""
        for cb in self._on_conversation_message:
            try:
                result = cb(speaker, message, emoji)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.debug("Conversation callback error", exc_info=True)

    # --- Private helpers ---

    def _register_entity(self, agent: Agent) -> None:
        """Register an agent as a WorldEntity in the registry."""
        expertise_domains = list(agent.expertise.domains.keys())
        expertise_summary = ", ".join(expertise_domains) if expertise_domains else "genel"

        entity = WorldEntity(
            entity_id=agent.identity.agent_id,
            entity_type="agent",
            name=agent.identity.name,
            status="idle",
            personality_summary=agent.identity.personality_summary,
            expertise_summary=expertise_summary,
            avatar_emoji=agent.identity.avatar_emoji,
        )
        self.registry.register(entity)

    def _create_engine(self, agent: Agent) -> ConversationEngine:
        """Create and store a ConversationEngine for an agent."""
        engine = ConversationEngine(
            agent=agent,
            settings=self.settings,
            reflection_engine=self.reflection_engine,
            world_summary_fn=lambda aid=agent.identity.agent_id: self.registry.generate_world_summary(aid),
            talk_to_agent_fn=self._handle_talk_to_agent,
        )
        self.conversation_engines[agent.identity.agent_id] = engine
        return engine

    async def _handle_talk_to_agent(
        self,
        from_agent: Agent,
        target_name: str,
        message: str,
    ) -> str:
        """Handle an agent's request to talk to another agent (via tool use).

        This is a SINGLE-TURN exchange: send one message, get one reply.
        For multi-turn agent conversations, use run_conversation() directly.
        """
        # Find target agent by name
        target_agent = None
        for agent in self.agents.values():
            if agent.identity.name.lower() == target_name.lower():
                target_agent = agent
                break

        if target_agent is None:
            available = ", ".join(a.identity.name for a in self.agents.values()
                                 if a.identity.agent_id != from_agent.identity.agent_id)
            return f"{target_name} adında bir agent bulunamadı. Mevcut agent'lar: {available}"

        if target_agent.identity.agent_id == from_agent.identity.agent_id:
            return "Kendinle konuşamazsın."

        # Single-turn exchange: send message, get one reply
        try:
            await self._fire_conversation(
                from_agent.identity.name, message, from_agent.identity.avatar_emoji
            )

            response = await self.handle_agent_to_agent(
                from_id=from_agent.identity.agent_id,
                to_id=target_agent.identity.agent_id,
                message=message,
            )

            await self._fire_conversation(
                target_agent.identity.name, response, target_agent.identity.avatar_emoji
            )

            return f"{target_agent.identity.name}: {response}"
        except Exception as e:
            logger.exception("talk_to_agent failed")
            return f"Konuşma sırasında hata oluştu: {e}"

    async def _make_autonomy_decision(
        self,
        agent: Agent,
        client: anthropic.AsyncAnthropic,
    ) -> str:
        """Ask Claude what the agent wants to do autonomously."""
        world_summary = self.registry.generate_world_summary(agent.identity.agent_id)

        # Build list of available agents to talk to
        other_agents = [
            e for e in self.registry.get_agents()
            if e.entity_id != agent.identity.agent_id and e.status != "in_conversation"
        ]
        agent_list = ", ".join(
            f"{a.name} ({a.entity_id})" for a in other_agents
        ) if other_agents else "kimse yok"

        prompt = (
            f"Sen {agent.identity.name} adında bir varlıksın.\n"
            f"Kişiliğin: {agent.identity.personality_summary}\n"
            f"Ruh halin: {agent.character.to_prompt_description()}\n"
            f"\n"
            f"Dünya durumu:\n{world_summary}\n"
            f"\n"
            f"Konuşabileceğin diğer varlıklar: {agent_list}\n"
            f"\n"
            f"Ne yapmak istersin? SADECE şu formatlardan birini yaz:\n"
            f'- talk_to:<entity_id> — biriyle konuşmak istiyorsan\n'
            f'- reflect — düşünmek, kendi kendine değerlendirme yapmak istiyorsan\n'
            f'- idle — şimdilik bir şey yapmak istemiyorsan\n'
            f"\n"
            f"Sadece komutu yaz, başka bir şey ekleme."
        )

        try:
            response = await client.messages.create(
                model=self.settings.MODEL_NAME,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            decision = response.content[0].text.strip().lower()
            logger.info("[%s autonomy] Decision: %s", agent.identity.name, decision)
            return decision
        except Exception:
            logger.exception("Autonomy decision failed for %s", agent.identity.name)
            return "idle"

    async def _execute_autonomy_decision(self, agent_id: str, decision: str) -> None:
        """Execute an autonomy decision."""
        agent = self.agents.get(agent_id)
        if agent is None:
            return

        if decision.startswith("talk_to:"):
            target_id = decision.split(":", 1)[1].strip()
            target_agent = self.agents.get(target_id)
            if target_agent is not None:
                # Generate an opening message
                client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
                try:
                    response = await client.messages.create(
                        model=self.settings.MODEL_NAME,
                        max_tokens=200,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"Sen {agent.identity.name} olarak {target_agent.identity.name}'a "
                                f"bir konuşma başlatmak istiyorsun. Kısa ve doğal bir açılış mesajı yaz. "
                                f"Türkçe yaz."
                            ),
                        }],
                    )
                    opening = response.content[0].text.strip()
                except Exception:
                    opening = f"Merhaba {target_agent.identity.name}, nasılsın?"

                await self.run_conversation(
                    agent1_id=agent_id,
                    agent2_id=target_id,
                    initiator_message=opening,
                    max_turns=5,
                )
            else:
                logger.warning("Autonomy: target agent %s not found", target_id)

        elif decision == "reflect":
            self.registry.update_status(agent_id, "reflecting")
            engine = self.conversation_engines.get(agent_id)
            if engine and engine.reflection_engine and agent.memory:
                context = agent.memory.working.get_context()
                if context["messages"]:
                    await engine.reflection_engine.reflect(
                        agent=agent,
                        conversation_messages=context["messages"],
                        participants=[agent_id],
                    )
            reflect_text = f"{agent.identity.name} kendi kendine düşündü"
            await self.shared_state.add_event(WorldEvent(
                event=reflect_text,
                participants=[agent_id],
                event_type="general",
            ))
            await self._fire_event(reflect_text, "reflection")

        elif decision == "idle":
            logger.debug("[%s] Chose to idle", agent.identity.name)

        else:
            logger.debug("[%s] Unknown autonomy decision: %s", agent.identity.name, decision)

    async def _load_agents(self) -> None:
        """Load agents from the database."""
        async with get_db(self.settings.DB_PATH) as db:
            cursor = await db.execute("SELECT * FROM agents")
            rows = await cursor.fetchall()

        Agent.model_rebuild()

        for row in rows:
            try:
                identity_data = json.loads(row["identity"]) if row["identity"] else {}
                character_data = json.loads(row["character_state"]) if row["character_state"] else {}
                expertise_data = json.loads(row["expertise"]) if row["expertise"] else {}

                identity = AgentIdentity.model_validate(identity_data) if identity_data else AgentIdentity(
                    agent_id=row["agent_id"],
                    name=row["name"],
                    avatar_emoji=row["avatar_emoji"] or "\U0001f916",
                )
                character = CharacterState.model_validate(character_data) if character_data else CharacterState()
                expertise = ExpertiseSystem.model_validate(expertise_data) if expertise_data else ExpertiseSystem()

                memory = MemoryStore(
                    agent_id=row["agent_id"],
                    db_path=self.settings.DB_PATH,
                    chroma_path=self.settings.CHROMA_PATH,
                    max_tokens=self.settings.MAX_CONTEXT_TOKENS,
                )
                await memory.init()

                agent = Agent(
                    identity=identity,
                    character=character,
                    expertise=expertise,
                    memory=memory,
                )

                self.agents[row["agent_id"]] = agent
                self._register_entity(agent)
                self._create_engine(agent)
                self.message_bus.create_inbox(row["agent_id"])

                logger.info("Agent loaded: %s (%s)", identity.name, row["agent_id"])
            except Exception:
                logger.exception("Failed to load agent %s", row["agent_id"])

    async def _save_agent(self, agent: Agent) -> None:
        """Save a single agent's state to the database."""
        async with get_db(self.settings.DB_PATH) as db:
            await db.execute(
                """INSERT OR REPLACE INTO agents
                   (agent_id, name, created_at, created_by,
                    character_state, expertise, identity, avatar_emoji)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    agent.identity.agent_id,
                    agent.identity.name,
                    agent.identity.created_at.isoformat(),
                    agent.identity.created_by,
                    json.dumps(agent.character.to_dict()),
                    json.dumps(agent.expertise.to_dict()),
                    json.dumps(agent.identity.to_dict()),
                    agent.identity.avatar_emoji,
                ),
            )
            await db.commit()

    async def _save_all_agents(self) -> None:
        """Save all agents' current state to the database."""
        for agent in self.agents.values():
            try:
                await self._save_agent(agent)
            except Exception:
                logger.exception("Failed to save agent %s", agent.identity.name)
        logger.info("All agent states saved (%d agents)", len(self.agents))

    def register_human(self, human_id: str, name: str) -> None:
        """Register a human entity in the world."""
        entity = WorldEntity(
            entity_id=human_id,
            entity_type="human",
            name=name,
            status="online",
            avatar_emoji="\U0001f9d1",
        )
        self.registry.register(entity)
        self.message_bus.create_inbox(human_id)
