"""ConversationEngine — async Claude API conversation loop.

Manages the full chat cycle: message intake, memory context retrieval,
prompt building, Claude API call with retries, working memory updates,
and reflection triggering at threshold.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import uuid4

import anthropic

from config.settings import Settings
from conversation.context_builder import build_messages, build_system_prompt
from core.token_tracker import TokenTracker

if TYPE_CHECKING:
    from conversation.reflection import ReflectionEngine
    from core.agent import Agent

logger = logging.getLogger(__name__)

# Exponential backoff config
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


AGENT_TOOLS = [
    {
        "name": "talk_to_agent",
        "description": (
            "Başka bir agent ile konuşma başlat. "
            "Bu tool'u kullanıcı senden başka bir agent'la konuşmanı, "
            "ona bir şey sormanı veya mesaj iletmeni istediğinde kullan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Konuşmak istediğin agent'ın adı",
                },
                "message": {
                    "type": "string",
                    "description": "Agent'a göndermek istediğin mesaj",
                },
            },
            "required": ["agent_name", "message"],
        },
    },
]


class ConversationEngine:
    """Drives conversations between an agent and other entities via Claude API."""

    def __init__(
        self,
        agent: Agent,
        settings: Settings | None = None,
        reflection_engine: ReflectionEngine | None = None,
        world_summary_fn: Callable[[], str] | None = None,
        talk_to_agent_fn: Callable | None = None,
    ):
        self.agent = agent
        self.settings = settings or Settings()
        self.client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        self.reflection_engine = reflection_engine
        self._world_summary_fn = world_summary_fn
        self._talk_to_agent_fn = talk_to_agent_fn

        # Conversation tracking
        self.conversation_id: str = str(uuid4())
        self.turn_count: int = 0
        self._last_reflection_turn: int = 0
        self.participants: set[str] = set()

        # Token limits: shorter for agent-to-agent, longer for human conversations
        self.max_tokens_human: int = 512
        self.max_tokens_agent: int = 256

    async def chat(self, user_message: str, sender_id: str = "human") -> str:
        """Process an incoming message and return the agent's response.

        Steps:
        1. Add message to working memory
        2. Build memory context from episodic/semantic recall
        3. Get world summary
        4. Build system prompt and messages
        5. Call Claude API with retries
        6. Add response to working memory
        7. Trigger reflection if threshold reached
        8. Return response
        """
        memory = self.agent.memory
        if memory is None:
            raise RuntimeError(f"Agent {self.agent.identity.name} has no memory initialized")

        # Track participants
        self.participants.add(sender_id)
        self.participants.add(self.agent.identity.agent_id)

        # 1. Add incoming message to working memory (with sender identity)
        sender_label = self._resolve_sender_name(sender_id)
        tagged_message = f"[{sender_label}]: {user_message}" if sender_label else user_message
        memory.working.add_message("user", tagged_message)

        # 2. Build memory context
        memory_context = await memory.build_memory_context(user_message)

        # 3. World summary
        world_summary = ""
        if self._world_summary_fn is not None:
            world_summary = self._world_summary_fn()

        # 4. Build prompt
        system_prompt = build_system_prompt(
            self.agent,
            memory_context=memory_context,
            world_summary=world_summary,
        )
        messages = build_messages(memory.working)

        # 5. Call Claude API (with tools if human conversation)
        is_agent = sender_id in self._get_agent_ids()
        use_tools = not is_agent and self._talk_to_agent_fn is not None
        max_tokens = self.max_tokens_agent if is_agent else self.max_tokens_human
        response_text = await self._call_claude(system_prompt, messages, use_tools=use_tools, max_tokens=max_tokens)

        # 6. Add response to working memory
        memory.working.add_message("assistant", response_text)
        self.turn_count += 1

        # 7. Compress working memory if needed
        await self._compress_context()

        # 8. Trigger reflection at threshold
        if (
            self.reflection_engine is not None
            and self.turn_count > 0
            and self.turn_count % self.settings.REFLECTION_THRESHOLD == 0
        ):
            await self._trigger_reflection()
            self._last_reflection_turn = self.turn_count

        return response_text

    async def end_conversation(self) -> None:
        """End the current conversation and trigger final reflection."""
        has_unreflected = self.turn_count > self._last_reflection_turn
        if self.reflection_engine is not None and has_unreflected:
            await self._trigger_reflection()

        # Store conversation record
        memory = self.agent.memory
        if memory is not None:
            from memory.database import get_db

            async with get_db(memory.db_path) as db:
                await db.execute(
                    """INSERT OR REPLACE INTO conversations
                       (conversation_id, participants, started_at, turn_count, summary)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        self.conversation_id,
                        json.dumps(list(self.participants)),
                        datetime.now(timezone.utc).isoformat(),
                        self.turn_count,
                        memory.working.summary or "",
                    ),
                )
                await db.commit()

        logger.info(
            "Conversation %s ended (%d turns)",
            self.conversation_id,
            self.turn_count,
        )

    def reset(self) -> None:
        """Reset for a new conversation."""
        if self.agent.memory is not None:
            self.agent.memory.working.clear()
        self.conversation_id = str(uuid4())
        self.turn_count = 0
        self._last_reflection_turn = 0
        self.participants.clear()

    def _get_agent_ids(self) -> set[str]:
        """Get set of known agent IDs (for detecting agent-to-agent vs human chat)."""
        if self._talk_to_agent_fn and hasattr(self._talk_to_agent_fn, '__self__'):
            orch = self._talk_to_agent_fn.__self__
            if hasattr(orch, 'agents'):
                return set(orch.agents.keys())
        return set()

    def _resolve_sender_name(self, sender_id: str) -> str:
        """Resolve a sender_id to a display name (e.g. 'Operator' or agent name)."""
        if self._talk_to_agent_fn and hasattr(self._talk_to_agent_fn, '__self__'):
            orch = self._talk_to_agent_fn.__self__
            if hasattr(orch, 'registry'):
                entity = orch.registry.get(sender_id)
                if entity:
                    return entity.name
        if sender_id == "human":
            return "Operator"
        return sender_id

    async def _call_claude(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        use_tools: bool = False,
        max_tokens: int = 512,
    ) -> str:
        """Call Claude API with exponential backoff retries and optional tool use."""
        last_error: Exception | None = None

        kwargs: dict[str, Any] = {
            "model": self.settings.MODEL_NAME,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if use_tools:
            kwargs["tools"] = AGENT_TOOLS

        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(
                    "Claude API call (attempt %d): %d messages, system prompt %d chars, tools=%s",
                    attempt + 1,
                    len(messages),
                    len(system_prompt),
                    use_tools,
                )
                response = await self.client.messages.create(**kwargs)
                TokenTracker().record(response.usage)

                # Handle tool use
                if response.stop_reason == "tool_use":
                    return await self._handle_tool_response(response, system_prompt, messages, kwargs)

                text = response.content[0].text
                logger.debug("Claude API response: %d chars", len(text))
                return text

            except anthropic.RateLimitError as e:
                last_error = e
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited, retrying in %.1fs (attempt %d)", delay, attempt + 1)
                await asyncio.sleep(delay)

            except anthropic.APITimeoutError as e:
                last_error = e
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("API timeout, retrying in %.1fs (attempt %d)", delay, attempt + 1)
                await asyncio.sleep(delay)

            except anthropic.APIError as e:
                last_error = e
                logger.error("Claude API error: %s", e)
                break

        raise RuntimeError(f"Claude API call failed after {MAX_RETRIES} retries: {last_error}")

    async def _handle_tool_response(
        self,
        response,
        system_prompt: str,
        messages: list[dict[str, str]],
        kwargs: dict[str, Any],
    ) -> str:
        """Handle a tool_use response from Claude, supporting chained tool calls."""
        MAX_TOOL_ROUNDS = 3  # prevent infinite tool call loops
        current_response = response
        current_messages = list(messages)

        for _round in range(MAX_TOOL_ROUNDS):
            # Extract text and tool calls from current response
            text_parts = []
            tool_results = []

            for block in current_response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use" and block.name == "talk_to_agent":
                    agent_name = block.input.get("agent_name", "")
                    message = block.input.get("message", "")
                    logger.info(
                        "[%s] Tool call: talk_to_agent(%s, %s)",
                        self.agent.identity.name, agent_name, message[:50],
                    )
                    result = await self._execute_talk_to_agent(agent_name, message)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # No tool calls — return collected text
            if not tool_results:
                return " ".join(text_parts) if text_parts else ""

            # Convert response content blocks to plain dicts for serialization
            assistant_content = []
            for block in current_response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_input = block.input
                    if hasattr(tool_input, "model_dump"):
                        tool_input = tool_input.model_dump()
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(tool_input) if tool_input else {},
                    })

            # Append assistant + tool_result turns and call Claude again
            current_messages = current_messages + [
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": tool_results},
            ]
            current_response = await self.client.messages.create(
                model=self.settings.MODEL_NAME,
                max_tokens=self.max_tokens_human,
                system=system_prompt,
                messages=current_messages,
                tools=AGENT_TOOLS,
            )
            TokenTracker().record(current_response.usage)

            # If this response is pure text, return it
            if current_response.stop_reason != "tool_use":
                return self._extract_text(current_response)

        # Exhausted rounds — return whatever text we have
        return self._extract_text(current_response)

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text content from a Claude response, ignoring tool_use blocks."""
        parts = [b.text for b in response.content if b.type == "text"]
        return " ".join(parts) if parts else ""

    async def _execute_talk_to_agent(self, agent_name: str, message: str) -> str:
        """Execute the talk_to_agent tool — triggers real agent conversation."""
        if self._talk_to_agent_fn is None:
            return "Konuşma başlatılamadı: sistem bağlantısı yok."

        try:
            result = await self._talk_to_agent_fn(
                from_agent=self.agent,
                target_name=agent_name,
                message=message,
            )
            return result
        except Exception as e:
            logger.exception("talk_to_agent failed")
            return f"Konuşma başlatılamadı: {e}"

    async def _compress_context(self) -> None:
        """Compress working memory if approaching token limit."""
        memory = self.agent.memory
        if memory is None:
            return

        async def summarize_fn(prompt: str) -> str:
            """Use Claude to summarize conversation for compression."""
            response = await self.client.messages.create(
                model=self.settings.MODEL_NAME,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            TokenTracker().record(response.usage)
            return response.content[0].text

        compressed = await memory.working.compress_if_needed(summarize_fn)
        if compressed:
            logger.info("Working memory compressed for agent %s", self.agent.identity.name)

    async def _trigger_reflection(self) -> None:
        """Trigger reflection engine on current conversation."""
        if self.reflection_engine is None:
            return

        memory = self.agent.memory
        if memory is None:
            return

        context = memory.working.get_context()
        try:
            await self.reflection_engine.reflect(
                agent=self.agent,
                conversation_messages=context["messages"],
                participants=list(self.participants),
                conversation_id=self.conversation_id,
            )
            logger.info(
                "Reflection completed for agent %s (turn %d)",
                self.agent.identity.name,
                self.turn_count,
            )
        except Exception:
            logger.exception("Reflection failed for agent %s", self.agent.identity.name)
