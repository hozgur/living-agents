from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentIdentity(BaseModel):
    """Core identity of an agent — immutable after creation."""

    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    personality_summary: str = ""
    avatar_emoji: str = "🤖"

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "AgentIdentity":
        return cls.model_validate(data)
