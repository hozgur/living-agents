"""MessageBus — async message passing between agents and humans.

Uses asyncio.Queue per entity for real-time delivery,
with SQLite persistence for message history.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from memory.database import get_db

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A message between entities."""

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    from_id: str
    to_id: str
    message_type: str = "chat"  # "chat" | "system" | "notification" | "request"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    requires_response: bool = False
    metadata: dict = Field(default_factory=dict)


class MessageBus:
    """Async message bus with per-entity queues and SQLite persistence."""

    def __init__(self, db_path: str = "data/agents.db"):
        self.db_path = db_path
        self._queues: dict[str, asyncio.Queue[Message]] = {}

    def create_inbox(self, entity_id: str) -> None:
        """Create a message queue for an entity."""
        if entity_id not in self._queues:
            self._queues[entity_id] = asyncio.Queue()
            logger.debug("Inbox created for %s", entity_id)

    def remove_inbox(self, entity_id: str) -> None:
        """Remove an entity's message queue."""
        self._queues.pop(entity_id, None)

    async def send(self, message: Message) -> None:
        """Send a message to the target entity's queue and persist to SQLite."""
        # Persist to database
        await self._persist_message(message)

        # Deliver to queue if recipient has an inbox
        queue = self._queues.get(message.to_id)
        if queue is not None:
            await queue.put(message)
            logger.debug(
                "Message %s: %s -> %s [%s]",
                message.message_id, message.from_id, message.to_id, message.message_type,
            )
        else:
            logger.debug(
                "Message %s persisted but no inbox for %s",
                message.message_id, message.to_id,
            )

    async def receive(
        self,
        entity_id: str,
        timeout: float | None = None,
    ) -> Message | None:
        """Receive the next message from an entity's queue.

        Returns None if timeout expires or no inbox exists.
        """
        queue = self._queues.get(entity_id)
        if queue is None:
            return None

        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                return queue.get_nowait()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None

    async def broadcast(
        self,
        from_id: str,
        content: str,
        msg_type: str = "notification",
    ) -> None:
        """Send a message to all entities with inboxes (except sender)."""
        for entity_id in self._queues:
            if entity_id != from_id:
                msg = Message(
                    from_id=from_id,
                    to_id=entity_id,
                    message_type=msg_type,
                    content=content,
                )
                await self.send(msg)

    def get_pending_count(self, entity_id: str) -> int:
        """Get the number of pending messages in an entity's queue."""
        queue = self._queues.get(entity_id)
        if queue is None:
            return 0
        return queue.qsize()

    async def get_history(
        self,
        entity_id: str,
        limit: int = 50,
    ) -> list[Message]:
        """Get recent message history for an entity from SQLite."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """SELECT * FROM messages
                   WHERE from_id = ? OR to_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (entity_id, entity_id, limit),
            )
            rows = await cursor.fetchall()
            return [self._row_to_message(row) for row in rows]

    async def _persist_message(self, message: Message) -> None:
        """Store a message in SQLite."""
        async with get_db(self.db_path) as db:
            await db.execute(
                """INSERT INTO messages
                   (message_id, from_id, to_id, message_type, content,
                    timestamp, requires_response, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.message_id,
                    message.from_id,
                    message.to_id,
                    message.message_type,
                    message.content,
                    message.timestamp.isoformat(),
                    message.requires_response,
                    json.dumps(message.metadata),
                ),
            )
            await db.commit()

    @staticmethod
    def _row_to_message(row) -> Message:
        """Convert a database row to a Message object."""
        return Message(
            message_id=row["message_id"],
            from_id=row["from_id"],
            to_id=row["to_id"],
            message_type=row["message_type"],
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            requires_response=bool(row["requires_response"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
