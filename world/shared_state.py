"""SharedWorldState — shared facts and events accessible to all agents.

Persists world facts and events to SQLite. Provides a Turkish summary
of recent world activity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from memory.database import get_db

logger = logging.getLogger(__name__)

# Valid event types
VALID_EVENT_TYPES = frozenset({
    "creation", "conversation", "discovery", "mood_change", "relationship_change", "general",
})


class WorldFact(BaseModel):
    """A fact known to the world, potentially confirmed by multiple entities."""

    fact_id: int | None = None  # auto-incremented by SQLite
    fact: str
    added_by: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confirmed_by: list[str] = Field(default_factory=list)


class WorldEvent(BaseModel):
    """An event that occurred in the world."""

    event_id: int | None = None  # auto-incremented by SQLite
    event: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    participants: list[str] = Field(default_factory=list)
    event_type: str = "general"


class SharedWorldState:
    """Manages shared world facts and events with SQLite persistence."""

    def __init__(self, db_path: str = "data/agents.db"):
        self.db_path = db_path

    async def add_fact(self, fact: str, added_by: str) -> WorldFact:
        """Add a new world fact."""
        wf = WorldFact(fact=fact, added_by=added_by)
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO world_facts (fact, added_by, timestamp, confirmed_by)
                   VALUES (?, ?, ?, ?)""",
                (wf.fact, wf.added_by, wf.timestamp.isoformat(), json.dumps(wf.confirmed_by)),
            )
            wf.fact_id = cursor.lastrowid
            await db.commit()
        logger.debug("World fact added by %s: %s", added_by, fact)
        return wf

    async def confirm_fact(self, fact_id: int, confirmed_by: str) -> None:
        """Add a confirming entity to a world fact."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                "SELECT confirmed_by FROM world_facts WHERE fact_id = ?",
                (fact_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            existing = json.loads(row["confirmed_by"]) if row["confirmed_by"] else []
            if confirmed_by not in existing:
                existing.append(confirmed_by)
                await db.execute(
                    "UPDATE world_facts SET confirmed_by = ? WHERE fact_id = ?",
                    (json.dumps(existing), fact_id),
                )
                await db.commit()

    async def add_event(self, event: WorldEvent) -> WorldEvent:
        """Record a world event."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO world_events (event, timestamp, participants, event_type)
                   VALUES (?, ?, ?, ?)""",
                (
                    event.event,
                    event.timestamp.isoformat(),
                    json.dumps(event.participants),
                    event.event_type if event.event_type in VALID_EVENT_TYPES else "general",
                ),
            )
            event.event_id = cursor.lastrowid
            await db.commit()
        logger.debug("World event: %s [%s]", event.event, event.event_type)
        return event

    async def get_recent_events(self, n: int = 20) -> list[WorldEvent]:
        """Get the most recent world events."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM world_events ORDER BY timestamp DESC LIMIT ?",
                (n,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_event(row) for row in rows]

    async def get_facts(self) -> list[WorldFact]:
        """Get all world facts."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM world_facts ORDER BY timestamp DESC",
            )
            rows = await cursor.fetchall()
            return [self._row_to_fact(row) for row in rows]

    async def to_summary(self, max_events: int = 10) -> str:
        """Generate a Turkish natural-language summary of world state."""
        parts = []

        # Recent events
        events = await self.get_recent_events(n=max_events)
        if events:
            parts.append("### Son Olaylar")
            for ev in events:
                type_label = _event_type_to_turkish(ev.event_type)
                parts.append(f"- [{type_label}] {ev.event}")

        # World facts
        facts = await self.get_facts()
        if facts:
            parts.append("### Dünya Gerçekleri")
            for wf in facts:
                confirmations = len(wf.confirmed_by)
                confirm_str = f" ({confirmations} doğrulama)" if confirmations > 0 else ""
                parts.append(f"- {wf.fact}{confirm_str}")

        return "\n".join(parts) if parts else "(Henüz dünya olayı yok)"

    @staticmethod
    def _row_to_event(row) -> WorldEvent:
        return WorldEvent(
            event_id=row["event_id"],
            event=row["event"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            participants=json.loads(row["participants"]) if row["participants"] else [],
            event_type=row["event_type"] or "general",
        )

    @staticmethod
    def _row_to_fact(row) -> WorldFact:
        return WorldFact(
            fact_id=row["fact_id"],
            fact=row["fact"],
            added_by=row["added_by"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            confirmed_by=json.loads(row["confirmed_by"]) if row["confirmed_by"] else [],
        )


def _event_type_to_turkish(event_type: str) -> str:
    return {
        "creation": "yaratılış",
        "conversation": "konuşma",
        "discovery": "keşif",
        "mood_change": "ruh hali",
        "relationship_change": "ilişki",
        "general": "genel",
    }.get(event_type, event_type)
