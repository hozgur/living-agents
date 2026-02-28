"""Episodic memory — stores and recalls conversation episodes."""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import chromadb
from pydantic import BaseModel, Field

from memory.database import get_db

logger = logging.getLogger(__name__)

# Emotional tones considered "intense" — decay slower
INTENSE_EMOTIONS = frozenset({
    "öfke", "korku", "heyecan", "şaşkınlık", "hayranlık",
    "üzüntü", "sevinç", "anger", "fear", "excitement",
    "surprise", "awe", "sadness", "joy",
})


class Episode(BaseModel):
    """A single episodic memory entry."""

    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    participants: list[str] = Field(default_factory=list)
    summary: str
    emotional_tone: str = "nötr"
    key_facts: list[str] = Field(default_factory=list)
    importance: float = 0.5
    current_importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    conversation_id: str | None = None


class EpisodicMemory:
    """Manages episodic memories for a single agent using SQLite + ChromaDB."""

    def __init__(self, agent_id: str, db_path: str, chroma_path: str):
        self.agent_id = agent_id
        self.db_path = db_path
        self.chroma_path = chroma_path
        self._collection = None

    async def init(self) -> None:
        """Initialize ChromaDB collection for this agent."""
        client = chromadb.PersistentClient(path=self.chroma_path)
        collection_name = f"agent-{self.agent_id}-episodes"
        # ChromaDB collection names: 3-63 chars, alphanumeric/hyphens/underscores
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
        self._collection = client.get_or_create_collection(name=collection_name)
        logger.debug("ChromaDB collection initialized: %s", collection_name)

    async def add_episode(self, episode: Episode) -> None:
        """Store an episode in SQLite and add its embedding to ChromaDB."""
        async with get_db(self.db_path) as db:
            await db.execute(
                """INSERT INTO episodes
                   (episode_id, agent_id, timestamp, participants, summary,
                    emotional_tone, key_facts, importance, current_importance,
                    tags, conversation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode.episode_id,
                    episode.agent_id,
                    episode.timestamp.isoformat(),
                    json.dumps(episode.participants),
                    episode.summary,
                    episode.emotional_tone,
                    json.dumps(episode.key_facts),
                    episode.importance,
                    episode.current_importance,
                    json.dumps(episode.tags),
                    episode.conversation_id,
                ),
            )
            await db.commit()

        # Add to ChromaDB for similarity search
        if self._collection is not None:
            self._collection.add(
                documents=[episode.summary],
                ids=[episode.episode_id],
                metadatas=[{
                    "agent_id": episode.agent_id,
                    "emotional_tone": episode.emotional_tone,
                    "importance": episode.importance,
                    "timestamp": episode.timestamp.isoformat(),
                }],
            )
        logger.debug("Episode stored: %s", episode.episode_id)

    async def recall(self, query: str, n: int = 5) -> list[Episode]:
        """Recall episodes similar to query using ChromaDB similarity search."""
        if self._collection is None or self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._collection.count()),
        )

        episode_ids = results["ids"][0] if results["ids"] else []
        if not episode_ids:
            return []

        return await self._fetch_episodes_by_ids(episode_ids)

    async def recall_about(self, entity_id: str, n: int = 5) -> list[Episode]:
        """Recall episodes involving a specific entity."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """SELECT * FROM episodes
                   WHERE agent_id = ? AND participants LIKE ?
                   ORDER BY current_importance DESC
                   LIMIT ?""",
                (self.agent_id, f'%"{entity_id}"%', n),
            )
            rows = await cursor.fetchall()
            return [self._row_to_episode(row) for row in rows]

    async def decay_memories(self, decay_rate: float = 0.01) -> None:
        """Apply importance decay to all episodes based on age and emotion."""
        now = datetime.now(timezone.utc)
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                "SELECT episode_id, timestamp, emotional_tone, current_importance FROM episodes WHERE agent_id = ?",
                (self.agent_id,),
            )
            rows = await cursor.fetchall()

            for row in rows:
                ts = datetime.fromisoformat(row["timestamp"]).replace(tzinfo=timezone.utc)
                days_elapsed = max((now - ts).total_seconds() / 86400, 0)
                emotion_modifier = (
                    0.5 if row["emotional_tone"] in INTENSE_EMOTIONS else 1.0
                )
                new_importance = max(
                    0.0,
                    row["current_importance"] - decay_rate * days_elapsed * emotion_modifier,
                )
                await db.execute(
                    "UPDATE episodes SET current_importance = ? WHERE episode_id = ?",
                    (new_importance, row["episode_id"]),
                )
            await db.commit()
        logger.debug("Memory decay applied for agent %s (%d episodes)", self.agent_id, len(rows))

    async def get_important_memories(self, threshold: float = 0.5) -> list[Episode]:
        """Get episodes with current_importance above threshold."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """SELECT * FROM episodes
                   WHERE agent_id = ? AND current_importance >= ?
                   ORDER BY current_importance DESC""",
                (self.agent_id, threshold),
            )
            rows = await cursor.fetchall()
            return [self._row_to_episode(row) for row in rows]

    async def forget(self, episode_id: str) -> None:
        """Delete an episode from both SQLite and ChromaDB."""
        async with get_db(self.db_path) as db:
            await db.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
            await db.commit()

        if self._collection is not None:
            try:
                self._collection.delete(ids=[episode_id])
            except Exception:
                logger.warning("Failed to delete episode %s from ChromaDB", episode_id)

        logger.debug("Episode forgotten: %s", episode_id)

    async def _fetch_episodes_by_ids(self, episode_ids: list[str]) -> list[Episode]:
        """Fetch full episode objects from SQLite by IDs."""
        if not episode_ids:
            return []
        placeholders = ",".join("?" for _ in episode_ids)
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
                episode_ids,
            )
            rows = await cursor.fetchall()
            return [self._row_to_episode(row) for row in rows]

    @staticmethod
    def _row_to_episode(row) -> Episode:
        """Convert a database row to an Episode object."""
        return Episode(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            participants=json.loads(row["participants"]) if row["participants"] else [],
            summary=row["summary"],
            emotional_tone=row["emotional_tone"] or "nötr",
            key_facts=json.loads(row["key_facts"]) if row["key_facts"] else [],
            importance=row["importance"],
            current_importance=row["current_importance"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            conversation_id=row["conversation_id"],
        )
