"""WorldRegistry — tracks all entities (agents and humans) in the world.

Singleton pattern. Provides perspective-based world summaries in Turkish.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Valid entity statuses
VALID_STATUSES = frozenset({
    "online", "offline", "idle", "thinking", "in_conversation", "reflecting",
})


class WorldEntity(BaseModel):
    """A registered entity in the world (agent or human)."""

    entity_id: str
    entity_type: str = "agent"  # "human" | "agent"
    name: str
    status: str = "idle"
    current_conversation_with: Optional[str] = None
    personality_summary: str = ""
    expertise_summary: str = ""
    avatar_emoji: str = "\U0001f916"
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorldRegistry:
    """Singleton registry tracking all entities in the world."""

    _instance: WorldRegistry | None = None

    def __new__(cls) -> WorldRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entities: dict[str, WorldEntity] = {}
            cls._instance._initialized = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    @property
    def entities(self) -> dict[str, WorldEntity]:
        return self._entities

    def register(self, entity: WorldEntity) -> None:
        """Register a new entity in the world."""
        self._entities[entity.entity_id] = entity
        logger.info(
            "Entity registered: %s (%s) [%s]",
            entity.name, entity.entity_id, entity.entity_type,
        )

    def unregister(self, entity_id: str) -> None:
        """Remove an entity from the world."""
        removed = self._entities.pop(entity_id, None)
        if removed:
            logger.info("Entity unregistered: %s (%s)", removed.name, entity_id)

    def update_status(
        self,
        entity_id: str,
        status: str,
        conversation_with: str | None = None,
    ) -> None:
        """Update an entity's status and conversation partner."""
        entity = self._entities.get(entity_id)
        if entity is None:
            logger.warning("Cannot update status: entity %s not found", entity_id)
            return
        if status in VALID_STATUSES:
            entity.status = status
        entity.current_conversation_with = conversation_with
        entity.last_active = datetime.now(timezone.utc)

    def get(self, entity_id: str) -> WorldEntity | None:
        """Get a specific entity by ID."""
        return self._entities.get(entity_id)

    def get_all(self) -> list[WorldEntity]:
        """Get all registered entities."""
        return list(self._entities.values())

    def get_agents(self) -> list[WorldEntity]:
        """Get only agent entities."""
        return [e for e in self._entities.values() if e.entity_type == "agent"]

    def get_humans(self) -> list[WorldEntity]:
        """Get only human entities."""
        return [e for e in self._entities.values() if e.entity_type == "human"]

    def get_online(self) -> list[WorldEntity]:
        """Get entities that are not offline."""
        return [e for e in self._entities.values() if e.status != "offline"]

    def generate_world_summary(self, perspective_of: str = "") -> str:
        """Generate a Turkish natural-language world summary.

        The perspective entity sees itself as 'sen', others by name.
        """
        if not self._entities:
            return "(Dünyada henüz kimse yok)"

        parts = []
        entity_descriptions = []

        for entity in self._entities.values():
            if entity.entity_id == perspective_of:
                name = "sen"
            else:
                name = f"{entity.avatar_emoji} {entity.name}"

            status_desc = _status_to_turkish(entity.status)

            if entity.current_conversation_with:
                partner = self._entities.get(entity.current_conversation_with)
                partner_name = partner.name if partner else entity.current_conversation_with
                if entity.entity_id == perspective_of:
                    status_desc = f"{partner_name} ile konuşuyorsun"
                else:
                    status_desc = f"{partner_name} ile konuşuyor"

            entity_descriptions.append(f"{name} ({status_desc})")

        entity_list = ", ".join(entity_descriptions)
        parts.append(f"Dünyada şu anda: {entity_list}.")

        # Count by type
        agent_count = len(self.get_agents())
        human_count = len(self.get_humans())
        online_count = len(self.get_online())
        parts.append(
            f"Toplam {agent_count} agent ve {human_count} insan var. "
            f"{online_count} varlık aktif."
        )

        return "\n".join(parts)

    def notify_all(self, event: str, exclude: str | None = None) -> list[str]:
        """Return list of entity IDs that should be notified of an event.

        Actual notification delivery is handled by the MessageBus.
        Returns the IDs so the caller can send messages via the bus.
        """
        return [
            eid for eid in self._entities
            if eid != exclude
        ]


def _status_to_turkish(status: str) -> str:
    """Convert entity status to Turkish description."""
    return {
        "online": "çevrimiçi",
        "offline": "çevrimdışı",
        "idle": "boşta",
        "thinking": "düşünüyor",
        "in_conversation": "konuşmada",
        "reflecting": "düşünceye dalmış",
    }.get(status, status)
