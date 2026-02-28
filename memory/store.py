"""MemoryStore — unified orchestrator composing all three memory layers."""

import logging
from datetime import datetime, timedelta, timezone

from memory.database import init_database
from memory.episodic import Episode, EpisodicMemory
from memory.semantic import KnowledgeFact, SemanticMemory
from memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# Episodes older than this with low importance get archived (deleted)
ARCHIVE_AGE_DAYS = 90
ARCHIVE_IMPORTANCE_THRESHOLD = 0.1


class MemoryStore:
    """Composes episodic, semantic, and working memory for a single agent."""

    def __init__(
        self,
        agent_id: str,
        db_path: str = "data/agents.db",
        chroma_path: str = "data/chroma",
        max_tokens: int = 8000,
    ):
        self.agent_id = agent_id
        self.db_path = db_path
        self.episodic = EpisodicMemory(agent_id, db_path, chroma_path)
        self.semantic = SemanticMemory(agent_id, db_path)
        self.working = WorkingMemory(max_tokens)

    async def init(self) -> None:
        """Initialize database tables and ChromaDB collection."""
        await init_database(self.db_path)
        await self.episodic.init()
        logger.info("MemoryStore initialized for agent %s", self.agent_id)

    async def build_memory_context(self, current_query: str) -> str:
        """Build the 'Hafızan' section for the system prompt.

        Recalls relevant episodic memories and semantic facts,
        combines them into Turkish text.
        """
        parts = []

        # Episodic recall
        episodes = await self.episodic.recall(current_query, n=5)
        if episodes:
            parts.append("### Hatırladığın Anılar")
            for ep in episodes:
                parts.append(
                    f"- [{ep.emotional_tone}] {ep.summary} "
                    f"(önem: {ep.current_importance:.1f})"
                )

        # Important persistent memories
        important = await self.episodic.get_important_memories(threshold=0.7)
        # Deduplicate with already-recalled episodes
        recalled_ids = {ep.episode_id for ep in episodes}
        important = [ep for ep in important if ep.episode_id not in recalled_ids]
        if important:
            parts.append("### Önemli Anılar")
            for ep in important[:3]:
                parts.append(f"- {ep.summary}")

        # Semantic facts — query about entities mentioned in the query
        words = current_query.split()
        all_facts: list[KnowledgeFact] = []
        seen_fact_ids: set[str] = set()
        for word in words:
            if len(word) >= 3:  # Skip short words
                facts = await self.semantic.get_all_facts_about(word)
                for fact in facts:
                    if fact.fact_id not in seen_fact_ids:
                        all_facts.append(fact)
                        seen_fact_ids.add(fact.fact_id)

        if all_facts:
            parts.append("### Bildiğin Gerçekler")
            parts.append(SemanticMemory.to_prompt_summary(all_facts))

        return "\n".join(parts) if parts else ""

    async def save_episode(self, episode: Episode) -> None:
        """Delegate episode storage to episodic memory."""
        await self.episodic.add_episode(episode)

    async def save_fact(self, fact: KnowledgeFact) -> None:
        """Delegate fact storage to semantic memory."""
        await self.semantic.add_fact(fact)

    async def daily_maintenance(self, decay_rate: float = 0.01) -> None:
        """Run daily maintenance: decay memories and archive old low-importance ones."""
        await self.episodic.decay_memories(decay_rate)

        # Archive (delete) very old, very low-importance episodes
        cutoff = datetime.now(timezone.utc) - timedelta(days=ARCHIVE_AGE_DAYS)
        low_importance = await self.episodic.get_important_memories(threshold=0.0)
        for ep in low_importance:
            if (
                ep.current_importance < ARCHIVE_IMPORTANCE_THRESHOLD
                and ep.timestamp < cutoff
            ):
                await self.episodic.forget(ep.episode_id)
                logger.debug("Archived old episode: %s", ep.episode_id)

        logger.info("Daily maintenance completed for agent %s", self.agent_id)
