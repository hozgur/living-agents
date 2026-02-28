"""Semantic memory — triple-store knowledge graph for agents."""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from memory.database import get_db

logger = logging.getLogger(__name__)


class KnowledgeFact(BaseModel):
    """A subject-predicate-object knowledge triple."""

    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source: str | None = None
    learned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_confirmed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticMemory:
    """Manages semantic knowledge facts for a single agent."""

    def __init__(self, agent_id: str, db_path: str):
        self.agent_id = agent_id
        self.db_path = db_path

    async def add_fact(self, fact: KnowledgeFact) -> None:
        """Insert a new knowledge fact."""
        async with get_db(self.db_path) as db:
            await db.execute(
                """INSERT INTO knowledge_facts
                   (fact_id, agent_id, subject, predicate, object,
                    confidence, source, learned_at, last_confirmed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.fact_id,
                    fact.agent_id,
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    fact.confidence,
                    fact.source,
                    fact.learned_at.isoformat(),
                    fact.last_confirmed.isoformat(),
                ),
            )
            await db.commit()
        logger.debug("Fact stored: %s -> %s -> %s", fact.subject, fact.predicate, fact.object)

    async def query_about(self, subject: str) -> list[KnowledgeFact]:
        """Get all facts where subject matches."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM knowledge_facts WHERE agent_id = ? AND subject = ? ORDER BY confidence DESC",
                (self.agent_id, subject),
            )
            rows = await cursor.fetchall()
            return [self._row_to_fact(row) for row in rows]

    async def query_relation(self, subject: str, predicate: str) -> list[KnowledgeFact]:
        """Get facts matching both subject and predicate."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """SELECT * FROM knowledge_facts
                   WHERE agent_id = ? AND subject = ? AND predicate = ?
                   ORDER BY confidence DESC""",
                (self.agent_id, subject, predicate),
            )
            rows = await cursor.fetchall()
            return [self._row_to_fact(row) for row in rows]

    async def update_confidence(self, fact_id: str, new_confidence: float) -> None:
        """Update the confidence score of a fact."""
        async with get_db(self.db_path) as db:
            await db.execute(
                """UPDATE knowledge_facts
                   SET confidence = ?, last_confirmed = ?
                   WHERE fact_id = ?""",
                (new_confidence, datetime.now(timezone.utc).isoformat(), fact_id),
            )
            await db.commit()

    async def get_all_facts_about(self, entity: str) -> list[KnowledgeFact]:
        """Get all facts where entity appears as subject or object."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """SELECT * FROM knowledge_facts
                   WHERE agent_id = ? AND (subject = ? OR object = ?)
                   ORDER BY confidence DESC""",
                (self.agent_id, entity, entity),
            )
            rows = await cursor.fetchall()
            return [self._row_to_fact(row) for row in rows]

    async def contradict(self, fact_id: str, new_fact: KnowledgeFact) -> None:
        """Lower confidence of old fact and insert the contradicting new fact."""
        async with get_db(self.db_path) as db:
            # Lower old fact's confidence
            await db.execute(
                "UPDATE knowledge_facts SET confidence = confidence * 0.3 WHERE fact_id = ?",
                (fact_id,),
            )
            # Insert new fact
            await db.execute(
                """INSERT INTO knowledge_facts
                   (fact_id, agent_id, subject, predicate, object,
                    confidence, source, learned_at, last_confirmed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_fact.fact_id,
                    new_fact.agent_id,
                    new_fact.subject,
                    new_fact.predicate,
                    new_fact.object,
                    new_fact.confidence,
                    new_fact.source,
                    new_fact.learned_at.isoformat(),
                    new_fact.last_confirmed.isoformat(),
                ),
            )
            await db.commit()
        logger.debug("Fact %s contradicted by %s", fact_id, new_fact.fact_id)

    @staticmethod
    def to_prompt_summary(facts: list[KnowledgeFact]) -> str:
        """Format facts as Turkish natural language for system prompt."""
        if not facts:
            return "(Bilinen gerçek yok)"

        lines = []
        for fact in facts:
            confidence_label = (
                "kesin" if fact.confidence >= 0.9
                else "güvenilir" if fact.confidence >= 0.7
                else "belirsiz" if fact.confidence >= 0.4
                else "şüpheli"
            )
            lines.append(f"- {fact.subject} {fact.predicate} {fact.object} [{confidence_label}]")
        return "\n".join(lines)

    @staticmethod
    def _row_to_fact(row) -> KnowledgeFact:
        """Convert a database row to a KnowledgeFact object."""
        return KnowledgeFact(
            fact_id=row["fact_id"],
            agent_id=row["agent_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            source=row["source"],
            learned_at=datetime.fromisoformat(row["learned_at"]),
            last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
        )
