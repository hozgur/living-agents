"""ReflectionEngine — post-conversation self-assessment.

Runs after every REFLECTION_THRESHOLD messages or when a conversation ends.
Sends the conversation to Claude for structured JSON reflection, then applies
the results: new episodes, character updates, relationship changes, knowledge facts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import anthropic

from config.settings import Settings
from memory.episodic import Episode
from memory.semantic import KnowledgeFact

if TYPE_CHECKING:
    from core.agent import Agent

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0

# The reflection prompt template (Turkish)
REFLECTION_PROMPT = """Sen {agent_name} olarak az önce şu konuşmayı yaptın:

{conversation_summary}

Şimdi bu deneyimi değerlendir ve SADECE aşağıdaki JSON formatında yanıtla (başka metin ekleme):
{{
  "episode": {{
    "summary": "Bu konuşmadan ne hatırlamalısın? (Türkçe kısa özet)",
    "emotional_tone": "konuşmanın duygusal tonu (tek kelime)",
    "key_facts": ["öğrenilen somut bilgiler"],
    "importance": 0.0-1.0,
    "tags": ["ilgili etiketler"]
  }},
  "character_updates": {{
    "mood_changes": {{"energy": 0.0, "happiness": 0.0, "anxiety": 0.0, "focus": 0.0, "excitement": 0.0}},
    "trait_nudges": {{}},
    "new_beliefs": [],
    "removed_beliefs": []
  }},
  "relationship_updates": {{
    "entity_id_here": {{
      "trust_delta": 0.0,
      "familiarity_delta": 0.0,
      "sentiment_delta": 0.0,
      "new_notes": []
    }}
  }},
  "new_knowledge": [
    {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.8}}
  ],
  "self_reflection": "Kendi kendine düşüncen (loglarda görünür, prompt'a eklenmez)"
}}

Kurallar:
- mood_changes değerleri -0.2 ile +0.2 arasında olmalı
- trait_nudges değerleri -0.02 ile +0.02 arasında olmalı (çok küçük!)
- importance 0.0 ile 1.0 arasında olmalı
- Tüm metin Türkçe olmalı
- SADECE geçerli JSON döndür, başka bir şey yazma"""


class ReflectionEngine:
    """Performs structured self-reflection after conversations."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)

    async def reflect(
        self,
        agent: Agent,
        conversation_messages: list[dict[str, str]],
        participants: list[str],
        conversation_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Run reflection on a conversation and apply results.

        Returns the parsed reflection dict, or None if reflection failed.
        """
        if not conversation_messages:
            return None

        # Build conversation summary for reflection prompt
        conversation_summary = self._format_conversation(conversation_messages)

        prompt = REFLECTION_PROMPT.format(
            agent_name=agent.identity.name,
            conversation_summary=conversation_summary,
        )

        # Call Claude for reflection
        raw_response = await self._call_claude(prompt)
        if raw_response is None:
            logger.warning("Reflection API call failed for agent %s", agent.identity.name)
            return None

        # Parse JSON response
        reflection = self._parse_reflection_json(raw_response)
        if reflection is None:
            logger.warning(
                "Failed to parse reflection JSON for agent %s, using fallback",
                agent.identity.name,
            )
            reflection = self._build_fallback_reflection(
                conversation_messages, participants
            )

        # Apply reflection results
        await self._apply_reflection(
            agent=agent,
            reflection=reflection,
            participants=participants,
            conversation_id=conversation_id,
        )

        return reflection

    async def _apply_reflection(
        self,
        agent: Agent,
        reflection: dict[str, Any],
        participants: list[str],
        conversation_id: str | None,
    ) -> None:
        """Apply all reflection results to agent state and memory."""
        memory = agent.memory
        if memory is None:
            logger.warning("Cannot apply reflection — agent has no memory")
            return

        # 1. Save episode to episodic memory
        episode_data = reflection.get("episode", {})
        episode = Episode(
            agent_id=agent.identity.agent_id,
            participants=participants,
            summary=episode_data.get("summary", "Konuşma yapıldı"),
            emotional_tone=episode_data.get("emotional_tone", "nötr"),
            key_facts=episode_data.get("key_facts", []),
            importance=self._clamp(episode_data.get("importance", 0.5), 0.0, 1.0),
            current_importance=self._clamp(episode_data.get("importance", 0.5), 0.0, 1.0),
            tags=episode_data.get("tags", []),
            conversation_id=conversation_id,
        )
        await memory.save_episode(episode)

        # 2. Apply character updates
        char_updates = reflection.get("character_updates", {})

        # Mood changes
        mood_changes = char_updates.get("mood_changes", {})
        clamped_mood = {
            k: self._clamp(v, -0.2, 0.2)
            for k, v in mood_changes.items()
            if isinstance(v, (int, float))
        }
        if clamped_mood:
            agent.character.update_mood(clamped_mood)

        # Trait nudges (max ±0.02 enforced by evolve_trait)
        trait_nudges = char_updates.get("trait_nudges", {})
        for trait, delta in trait_nudges.items():
            if isinstance(delta, (int, float)):
                agent.character.evolve_trait(trait, delta)

        # Beliefs
        for belief in char_updates.get("new_beliefs", []):
            if isinstance(belief, str) and belief:
                agent.character.add_belief(belief)
        for belief in char_updates.get("removed_beliefs", []):
            if isinstance(belief, str) and belief:
                agent.character.remove_belief(belief)

        # 3. Apply relationship updates
        rel_updates = reflection.get("relationship_updates", {})
        for entity_id, updates in rel_updates.items():
            if not isinstance(updates, dict):
                continue
            rel_changes = {}
            if "trust_delta" in updates and isinstance(updates["trust_delta"], (int, float)):
                # Get current trust and apply delta
                current_rel = agent.character.relationships.get(entity_id)
                current_trust = current_rel.trust if current_rel else 0.5
                rel_changes["trust"] = self._clamp(
                    current_trust + updates["trust_delta"], 0.0, 1.0
                )
            if "familiarity_delta" in updates and isinstance(updates["familiarity_delta"], (int, float)):
                current_rel = agent.character.relationships.get(entity_id)
                current_fam = current_rel.familiarity if current_rel else 0.0
                rel_changes["familiarity"] = self._clamp(
                    current_fam + updates["familiarity_delta"], 0.0, 1.0
                )
            if "sentiment_delta" in updates and isinstance(updates["sentiment_delta"], (int, float)):
                current_rel = agent.character.relationships.get(entity_id)
                current_sent = current_rel.sentiment if current_rel else 0.0
                rel_changes["sentiment"] = self._clamp(
                    current_sent + updates["sentiment_delta"], -1.0, 1.0
                )
            for note in updates.get("new_notes", []):
                if isinstance(note, str) and note:
                    rel_changes["notes"] = note  # update_relationship appends strings
            if rel_changes:
                agent.character.update_relationship(entity_id, rel_changes)

        # 4. Save new knowledge facts
        for fact_data in reflection.get("new_knowledge", []):
            if not isinstance(fact_data, dict):
                continue
            subject = fact_data.get("subject", "")
            predicate = fact_data.get("predicate", "")
            obj = fact_data.get("object", "")
            if subject and predicate and obj:
                fact = KnowledgeFact(
                    agent_id=agent.identity.agent_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=self._clamp(fact_data.get("confidence", 0.8), 0.0, 1.0),
                    source=f"reflection:{conversation_id or 'unknown'}",
                )
                await memory.save_fact(fact)

        # 5. Log self-reflection
        self_reflection = reflection.get("self_reflection", "")
        if self_reflection:
            logger.info(
                "[%s iç düşünce] %s",
                agent.identity.name,
                self_reflection,
            )

    async def _call_claude(self, prompt: str) -> str | None:
        """Call Claude for reflection with exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=self.settings.MODEL_NAME,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            except anthropic.RateLimitError as e:
                last_error = e
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("Reflection rate limited, retrying in %.1fs", delay)
                await asyncio.sleep(delay)

            except (anthropic.APITimeoutError, anthropic.APIError) as e:
                last_error = e
                logger.warning("Reflection API error: %s", e)
                break

        logger.error("Reflection API call failed: %s", last_error)
        return None

    @staticmethod
    def _parse_reflection_json(raw_text: str) -> dict[str, Any] | None:
        """Parse reflection JSON from Claude's response.

        Handles cases where Claude wraps JSON in markdown code blocks.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove first line (```json or ```)
            lines = text.split("\n")
            lines = lines[1:]  # Remove opening fence
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
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
    def _format_conversation(messages: list[dict[str, str]]) -> str:
        """Format conversation messages as readable text for the reflection prompt."""
        lines = []
        for msg in messages:
            role = "Kullanıcı" if msg["role"] == "user" else "Sen"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    @staticmethod
    def _build_fallback_reflection(
        messages: list[dict[str, str]],
        participants: list[str],
    ) -> dict[str, Any]:
        """Build a minimal reflection when JSON parsing fails."""
        # Create a basic summary from the last few messages
        recent = messages[-3:] if len(messages) > 3 else messages
        summary_parts = [msg["content"][:100] for msg in recent]
        summary = "Konuşma yapıldı: " + " | ".join(summary_parts)

        return {
            "episode": {
                "summary": summary[:300],
                "emotional_tone": "nötr",
                "key_facts": [],
                "importance": 0.3,
                "tags": [],
            },
            "character_updates": {
                "mood_changes": {},
                "trait_nudges": {},
                "new_beliefs": [],
                "removed_beliefs": [],
            },
            "relationship_updates": {},
            "new_knowledge": [],
            "self_reflection": "Reflection JSON parse edilemedi, fallback kullanıldı.",
        }

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))
