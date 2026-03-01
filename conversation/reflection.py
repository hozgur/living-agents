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
from core.token_tracker import TokenTracker
from memory.episodic import Episode
from memory.semantic import KnowledgeFact

if TYPE_CHECKING:
    from core.agent import Agent

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0

# The reflection prompt template — English structure, {language} for text output
REFLECTION_PROMPT = """As {agent_name}, you just had this conversation:

{conversation_summary}

Participants: {participants_info}
Your current beliefs: {current_beliefs}

Now evaluate this experience and respond ONLY in the following JSON format (add no other text).
Write all text fields in {language}.
{{
  "episode": {{
    "summary": "WHO said WHAT and what did YOU think? Do NOT write generic summaries like 'we had a chat'. Be specific: what topic was discussed, who claimed what, which idea was interesting, what did you learn? Write concrete details you can recall in the next conversation.",
    "emotional_tone": "emotional tone of the conversation (single word)",
    "key_facts": ["each one a concrete and specific piece of info: person's name + what they said/did"],
    "importance": 0.0-1.0,
    "tags": ["relevant tags"],
    "follow_up": "What should you talk about with this person next time? Which topic could you explore deeper?"
  }},
  "character_updates": {{
    "mood_changes": {{"energy": 0.0, "happiness": 0.0, "anxiety": 0.0, "focus": 0.0, "excitement": 0.0}},
    "trait_nudges": {{}},
    "new_beliefs": ["a new belief born from this conversation, if any"],
    "removed_beliefs": ["something you no longer believe, if any"],
    "belief_evolutions": {{
      "existing belief text": 0.05,
      "another belief": -0.05
    }},
    "belief_transformations": {{
      "old belief text": "the evolved new version of this belief"
    }}
  }},
  "relationship_updates": {{
    "person_name": {{
      "trust_delta": 0.0,
      "familiarity_delta": 0.0,
      "sentiment_delta": 0.0,
      "new_notes": ["a concrete thing you learned about this person"]
    }}
  }},
  "new_knowledge": [
    {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.8}}
  ],
  "self_reflection": "How did this conversation affect you? Did your thoughts change?"
}}

Rules:
- mood_changes values must be between -0.2 and +0.2
- trait_nudges values must be between -0.02 and +0.02 (very small!)
- importance must be between 0.0 and 1.0
- All text MUST be in {language}
- Return ONLY valid JSON, nothing else
- NEVER write generic phrases like "we had a brief chat" in the summary. Give CONCRETE details.
- relationship_updates keys must be person NAMES (e.g., "Operator", "Luna", "Genesis")
- Each item in key_facts must follow "Who: what" format (e.g., "Luna: argues that truth is layered")
- BELIEF EVOLUTION: Look at your current beliefs. Did this conversation strengthen one (+0.05) or weaken one (-0.05)?
  If a belief transformed, write it in belief_transformations.
  belief_evolutions and belief_transformations can be empty but review your beliefs every reflection."""


class ReflectionEngine:
    """Performs structured self-reflection after conversations."""

    def __init__(self, settings: Settings | None = None, on_reflection_event=None):
        self.settings = settings or Settings()
        self.client = anthropic.AsyncAnthropic(api_key=self.settings.ANTHROPIC_API_KEY)
        # Callback: (agent_name, event_text, event_type) for UI event log
        self._on_reflection_event = on_reflection_event

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

        # Build participant info from conversation messages (they contain [Name]: tags)
        participants_info = self._extract_participant_names(
            conversation_messages, agent.identity.name
        )

        # Format current beliefs for the prompt
        current_beliefs = "none"
        if agent.character.beliefs:
            current_beliefs = "; ".join(
                f"'{b.text}' (strength: {b.conviction:.1f})"
                for b in agent.character.beliefs
            )

        prompt = REFLECTION_PROMPT.format(
            agent_name=agent.identity.name,
            conversation_summary=conversation_summary,
            participants_info=participants_info,
            current_beliefs=current_beliefs,
            language=self.settings.CHAT_LANGUAGE,
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

    async def _emit(self, agent_name: str, text: str, event_type: str = "reflection") -> None:
        """Fire a reflection event to the UI."""
        if self._on_reflection_event:
            try:
                result = self._on_reflection_event(agent_name, text, event_type)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

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

        name = agent.identity.name

        # 1. Save episode to episodic memory
        episode_data = reflection.get("episode", {})
        summary = episode_data.get("summary", "Conversation held")
        follow_up = episode_data.get("follow_up", "")
        if follow_up:
            summary = f"{summary} [Next time: {follow_up}]"
        episode = Episode(
            agent_id=agent.identity.agent_id,
            participants=participants,
            summary=summary,
            emotional_tone=episode_data.get("emotional_tone", "neutral"),
            key_facts=episode_data.get("key_facts", []),
            importance=self._clamp(episode_data.get("importance", 0.5), 0.0, 1.0),
            current_importance=self._clamp(episode_data.get("importance", 0.5), 0.0, 1.0),
            tags=episode_data.get("tags", []),
            conversation_id=conversation_id,
        )
        await memory.save_episode(episode)
        await self._emit(name, f"💾 New memory: {summary[:80]}", "memory")

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

        # Beliefs — add/remove
        for belief in char_updates.get("new_beliefs", []):
            if isinstance(belief, str) and belief:
                agent.character.add_belief(belief)
                await self._emit(name, f"🌱 New belief: \"{belief}\"", "belief")
        for belief in char_updates.get("removed_beliefs", []):
            if isinstance(belief, str) and belief:
                agent.character.remove_belief(belief)
                await self._emit(name, f"❌ Belief dropped: \"{belief}\"", "belief")

        # Belief evolutions — strengthen or weaken existing beliefs
        for belief_text, delta in char_updates.get("belief_evolutions", {}).items():
            if isinstance(delta, (int, float)) and isinstance(belief_text, str):
                agent.character.evolve_belief(belief_text, delta)
                direction = "strengthened 📈" if delta > 0 else "weakened 📉"
                await self._emit(
                    name, f"💭 Belief {direction}: \"{belief_text}\" ({delta:+.2f})", "belief"
                )

        # Belief transformations — old belief becomes new belief
        for old_text, new_text in char_updates.get("belief_transformations", {}).items():
            if isinstance(old_text, str) and isinstance(new_text, str) and new_text:
                agent.character.transform_belief(old_text, new_text)
                await self._emit(
                    name, f"🔄 Belief transformed: \"{old_text}\" → \"{new_text}\"", "belief"
                )

        # 3. Apply relationship updates (keyed by name, e.g. "Luna", "Operator")
        rel_updates = reflection.get("relationship_updates", {})
        for entity_id, updates in rel_updates.items():
            # Skip placeholder keys from the template
            if entity_id in ("entity_id_here", "kişi_adı"):
                continue
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
                await self._emit(
                    name, f"🤝 Relationship with {entity_id} updated", "relationship"
                )

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
                await self._emit(
                    name, f"📚 Learned: {subject} → {predicate} → {obj}", "knowledge"
                )

        # 5. Log self-reflection
        self_reflection = reflection.get("self_reflection", "")
        if self_reflection:
            logger.info(
                "[%s inner thought] %s",
                agent.identity.name,
                self_reflection,
            )

    @staticmethod
    def _extract_participant_names(
        messages: list[dict[str, str]], agent_name: str
    ) -> str:
        """Extract participant names from tagged messages like '[Name]: ...'."""
        import re
        names = {agent_name}
        tag_pattern = re.compile(r"^\[([^\]]+)\]:")
        for msg in messages:
            if msg["role"] == "user":
                match = tag_pattern.match(msg["content"])
                if match:
                    names.add(match.group(1))
        return ", ".join(sorted(names))

    async def _call_claude(self, prompt: str) -> str | None:
        """Call Claude for reflection with exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.messages.create(
                    model=self.settings.MODEL_REFLECTION,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                TokenTracker().record(response.usage)
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
        """Format conversation messages as readable text for the reflection prompt.

        User messages may contain sender tags like '[Operator]: ...' or
        '[AgentName]: ...' — preserve these so the reflection knows WHO spoke.
        """
        lines = []
        for msg in messages:
            if msg["role"] == "assistant":
                lines.append(f"Sen: {msg['content']}")
            else:
                # User messages already tagged as [SenderName]: ... — use as-is
                lines.append(msg["content"])
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
        summary = "Conversation held: " + " | ".join(summary_parts)

        return {
            "episode": {
                "summary": summary[:300],
                "emotional_tone": "neutral",
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
            "self_reflection": "Reflection JSON could not be parsed, fallback used.",
        }

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(max_val, value))
