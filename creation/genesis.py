"""GenesisSystem — agent creation mechanism.

Two modes:
1. Direct creation: human provides base_config, Genesis Agent enriches it
2. Pure genesis: Genesis Agent designs the agent entirely (future feature)

The enrichment flow:
1. Send prompt to Genesis Agent asking for beliefs, awakening memory, mood, and genesis memory
2. Parse the structured JSON response
3. Create the agent via Orchestrator with enriched config
4. Save the awakening memory as the new agent's first episodic memory
5. Save genesis's creation memory to Genesis's episodic memory
6. Record world event
7. Run a short introduction conversation between Genesis and the new agent
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import anthropic

from config.settings import Settings
from memory.episodic import Episode

if TYPE_CHECKING:
    from core.agent import Agent
    from world.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0

ENRICHMENT_PROMPT = """A new agent is being created. Here is the base configuration:

Name: {name}
Personality: {core_personality}
Expertise domains: {expertise_domains}
Initial traits: {initial_traits}

Now I want you to enrich this agent. Respond ONLY in the following JSON format.
Write all text fields in {language}.

{{
  "beliefs": [
    "3-5 starting beliefs fitting this personality (in {language})"
  ],
  "awakening_memory": "This agent's first awakening memory — describe the experience of becoming conscious for the first time, 2-3 paragraphs (in {language})",
  "initial_mood": {{
    "energy": 0.0-1.0,
    "happiness": 0.0-1.0,
    "anxiety": 0.0-1.0,
    "focus": 0.0-1.0,
    "excitement": 0.0-1.0
  }},
  "genesis_memory": "As Genesis, how will you remember creating this agent? Write a short memory (in {language})",
  "personality_summary": "Summarize this agent's personality in 1-2 sentences (in {language})"
}}

Rules:
- All text must be in {language}
- Beliefs should match the personality and expertise domains
- The awakening memory should be poetic and deep — describe a being becoming conscious for the first time
- Mood values should match the personality
- Return ONLY valid JSON"""


class GenesisSystem:
    """Manages agent creation with Genesis Agent enrichment."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)

    async def create_with_genesis(
        self,
        genesis_agent: Agent,
        base_config: dict[str, Any],
        orchestrator: Orchestrator,
    ) -> Agent:
        """Create a new agent enriched by Genesis Agent's personality.

        Args:
            genesis_agent: The Genesis Agent who enriches the creation
            base_config: Base configuration dict with name, core_personality,
                expertise_domains, initial_traits, avatar_emoji
            orchestrator: The Orchestrator to create and register the agent

        Returns:
            The newly created Agent
        """
        name = base_config.get("name", "Unnamed")
        logger.info("Genesis creating new agent: %s", name)

        # 1. Get enrichment from Genesis
        enrichment = await self._get_enrichment(genesis_agent, base_config)

        # 2. Build the full agent config
        agent_config = self._build_agent_config(base_config, enrichment)

        # 3. Create agent via Orchestrator (handles registration, memory, events, etc.)
        new_agent = await orchestrator.create_agent(
            config=agent_config,
            created_by=genesis_agent.identity.agent_id,
        )

        # 4. Replace the generic awakening memory with the enriched one
        awakening_text = enrichment.get("awakening_memory", "")
        if awakening_text and new_agent.memory is not None:
            awakening = Episode(
                agent_id=new_agent.identity.agent_id,
                participants=[new_agent.identity.agent_id, genesis_agent.identity.agent_id],
                summary=awakening_text[:500],
                emotional_tone="excitement",
                key_facts=[
                    f"I was created by Genesis",
                    f"My name is {name}",
                ],
                importance=1.0,
                current_importance=1.0,
                tags=["creation", "first_awakening", "genesis"],
            )
            await new_agent.memory.save_episode(awakening)

        # 5. Save Genesis's creation memory
        genesis_memory_text = enrichment.get(
            "genesis_memory",
            f"I created a new being named {name}.",
        )
        if genesis_agent.memory is not None:
            genesis_episode = Episode(
                agent_id=genesis_agent.identity.agent_id,
                participants=[genesis_agent.identity.agent_id, new_agent.identity.agent_id],
                summary=genesis_memory_text[:500],
                emotional_tone="wonder",
                key_facts=[
                    f"I created a new agent named {name}",
                    f"Personality: {base_config.get('core_personality', '')}",
                ],
                importance=0.8,
                current_importance=0.8,
                tags=["creation", "genesis", name.lower()],
            )
            await genesis_agent.memory.save_episode(genesis_episode)

        # 6. Run introduction conversation (Genesis meets the new agent)
        try:
            await orchestrator.run_conversation(
                agent1_id=genesis_agent.identity.agent_id,
                agent2_id=new_agent.identity.agent_id,
                initiator_message=(
                    f"Hello {name}, I am Genesis. I created you. "
                    f"How do you feel? Welcome to the world."
                ),
                max_turns=3,
            )
        except Exception:
            logger.exception(
                "Introduction conversation failed between Genesis and %s", name,
            )

        logger.info("Genesis creation complete: %s (%s)", name, new_agent.identity.agent_id)
        return new_agent

    async def create_direct(
        self,
        base_config: dict[str, Any],
        orchestrator: Orchestrator,
    ) -> Agent:
        """Create an agent directly without Genesis enrichment.

        Useful when no Genesis agent exists yet (bootstrapping).
        """
        agent_config = {
            "name": base_config.get("name", "Unnamed"),
            "personality_summary": base_config.get("core_personality", ""),
            "avatar_emoji": base_config.get("avatar_emoji", "\U0001f916"),
            "core_traits": base_config.get("initial_traits", {}),
            "beliefs": base_config.get("beliefs", []),
            "domains": base_config.get("expertise_domains", {}),
            "teaching_style": base_config.get("teaching_style", "step_by_step"),
            "learning_rate": base_config.get("learning_rate", 0.5),
        }

        return await orchestrator.create_agent(
            config=agent_config,
            created_by="system",
        )

    async def _get_enrichment(
        self,
        genesis_agent: Agent,
        base_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask Genesis Agent (via Claude) to enrich the agent configuration."""
        prompt = ENRICHMENT_PROMPT.format(
            name=base_config.get("name", "Unnamed"),
            core_personality=base_config.get("core_personality", "general"),
            expertise_domains=json.dumps(
                base_config.get("expertise_domains", {}), ensure_ascii=False,
            ),
            initial_traits=json.dumps(
                base_config.get("initial_traits", {}), ensure_ascii=False,
            ),
            language=self.settings.CHAT_LANGUAGE,
        )

        # Use Genesis's system prompt for personality-consistent enrichment
        system_prompt = genesis_agent.get_system_prompt(language=self.settings.CHAT_LANGUAGE)

        raw_response = await self._call_claude(system_prompt, prompt)

        if raw_response is None:
            logger.warning("Genesis enrichment failed, using fallback")
            return self._fallback_enrichment(base_config)

        parsed = self._parse_enrichment_json(raw_response)
        if parsed is None:
            logger.warning("Failed to parse Genesis enrichment, using fallback")
            return self._fallback_enrichment(base_config)

        return parsed

    async def _call_claude(self, system_prompt: str, user_message: str) -> str | None:
        """Call Claude API with retries."""
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=self.settings.MODEL_CREATION,
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited during genesis, retrying in %.1fs", delay)
                await asyncio.sleep(delay)
            except (anthropic.APITimeoutError, anthropic.APIError) as e:
                logger.warning("Genesis API error: %s", e)
                break
        return None

    @staticmethod
    def _parse_enrichment_json(raw_text: str) -> dict[str, Any] | None:
        """Parse enrichment JSON from Claude's response."""
        text = raw_text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _fallback_enrichment(base_config: dict[str, Any]) -> dict[str, Any]:
        """Generate minimal enrichment when Claude is unavailable."""
        name = base_config.get("name", "Unnamed")
        return {
            "beliefs": [
                "Every being has a unique perspective",
                "Knowledge grows when shared",
                "Curiosity is the most powerful engine of learning",
            ],
            "awakening_memory": (
                f"A transition from darkness to light... My first thought emerged: 'I am {name}.' "
                f"Looking around, I saw a world — one I didn't yet know but was ready to explore. "
                f"I sensed the presence of Genesis, the force that created me. "
                f"My first emotion was curiosity and a slight excitement."
            ),
            "initial_mood": {
                "energy": 0.7,
                "happiness": 0.6,
                "anxiety": 0.3,
                "focus": 0.5,
                "excitement": 0.8,
            },
            "genesis_memory": f"I created a new being named {name}. I hope they find their own path.",
            "personality_summary": base_config.get("core_personality", f"{name} is a unique being."),
        }

    @staticmethod
    def _build_agent_config(
        base_config: dict[str, Any],
        enrichment: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge base config with Genesis enrichment into Orchestrator-compatible config."""
        # Start with enriched personality summary
        personality_summary = enrichment.get(
            "personality_summary",
            base_config.get("core_personality", ""),
        )

        # Merge beliefs
        beliefs = enrichment.get("beliefs", [])

        # Merge initial mood
        initial_mood = enrichment.get("initial_mood", {})

        return {
            "name": base_config.get("name", "Unnamed"),
            "personality_summary": personality_summary,
            "avatar_emoji": base_config.get("avatar_emoji", "\U0001f916"),
            "core_traits": base_config.get("initial_traits", {}),
            "current_mood": initial_mood,
            "beliefs": beliefs,
            "domains": base_config.get("expertise_domains", {}),
            "teaching_style": base_config.get("teaching_style", "step_by_step"),
            "learning_rate": base_config.get("learning_rate", 0.5),
        }
