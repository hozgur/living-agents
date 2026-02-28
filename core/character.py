from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class RelationshipState(BaseModel):
    """Tracks an agent's relationship with another entity."""

    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    familiarity: float = Field(default=0.0, ge=0.0, le=1.0)
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    shared_experience_count: int = 0
    last_interaction: Optional[datetime] = None
    notes: list[str] = Field(default_factory=list)


class Belief(BaseModel):
    """A belief with conviction strength that can evolve over time."""

    text: str
    conviction: float = Field(default=0.7, ge=0.0, le=1.0)


class CharacterState(BaseModel):
    """Evolving personality state of an agent."""

    core_traits: dict[str, float] = Field(default_factory=lambda: {
        "curiosity": 0.5,
        "warmth": 0.5,
        "assertiveness": 0.5,
        "humor": 0.5,
        "patience": 0.5,
        "creativity": 0.5,
    })
    current_mood: dict[str, float] = Field(default_factory=lambda: {
        "energy": 0.5,
        "happiness": 0.5,
        "anxiety": 0.2,
        "focus": 0.5,
        "excitement": 0.3,
    })
    beliefs: list[Belief] = Field(default_factory=list)
    relationships: dict[str, RelationshipState] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_beliefs(cls, data: Any) -> Any:
        """Migrate old plain-string beliefs to Belief objects."""
        if isinstance(data, dict) and "beliefs" in data:
            migrated = []
            for b in data["beliefs"]:
                if isinstance(b, str):
                    migrated.append({"text": b, "conviction": 0.7})
                elif isinstance(b, dict):
                    migrated.append(b)
                else:
                    migrated.append(b)
            data["beliefs"] = migrated
        return data

    def update_mood(self, changes: dict[str, float]) -> None:
        """Update mood values, clamped to [0.0, 1.0]."""
        for key, delta in changes.items():
            if key in self.current_mood:
                self.current_mood[key] = max(0.0, min(1.0, self.current_mood[key] + delta))

    def evolve_trait(self, trait: str, delta: float) -> None:
        """Evolve a core trait by delta, clamped to max ±0.02 per call and [0.0, 1.0] range."""
        if trait not in self.core_traits:
            return
        clamped_delta = max(-0.02, min(0.02, delta))
        new_value = self.core_traits[trait] + clamped_delta
        self.core_traits[trait] = max(0.0, min(1.0, new_value))

    def update_relationship(self, entity_id: str, updates: dict) -> None:
        """Update or create a relationship with an entity."""
        if entity_id not in self.relationships:
            self.relationships[entity_id] = RelationshipState()

        rel = self.relationships[entity_id]
        for key, value in updates.items():
            if key == "notes" and isinstance(value, str):
                rel.notes.append(value)
            elif hasattr(rel, key):
                setattr(rel, key, value)
        rel.last_interaction = datetime.now(timezone.utc)

    def add_belief(self, belief: str, conviction: float = 0.7) -> None:
        """Add a new belief or strengthen an existing one."""
        for b in self.beliefs:
            if b.text == belief:
                # Belief already exists — strengthen it
                b.conviction = min(1.0, b.conviction + 0.05)
                return
        self.beliefs.append(Belief(text=belief, conviction=max(0.1, min(1.0, conviction))))

    def remove_belief(self, belief: str) -> None:
        """Remove a belief by text."""
        self.beliefs = [b for b in self.beliefs if b.text != belief]

    def evolve_belief(self, belief_text: str, delta: float) -> None:
        """Shift a belief's conviction by delta (clamped to ±0.1 per call).

        If conviction drops below 0.1, the belief is removed.
        """
        clamped = max(-0.1, min(0.1, delta))
        for b in self.beliefs:
            if b.text == belief_text:
                b.conviction = max(0.0, min(1.0, b.conviction + clamped))
                if b.conviction < 0.1:
                    self.beliefs.remove(b)
                return

    def transform_belief(self, old_text: str, new_text: str) -> None:
        """Transform a belief into a new version, carrying over conviction."""
        for b in self.beliefs:
            if b.text == old_text:
                b.text = new_text
                return
        # If old belief not found, add the new one
        self.add_belief(new_text, conviction=0.5)

    def to_prompt_description(self) -> str:
        """Generate Turkish natural language description for system prompt."""
        lines = []

        # Trait descriptions
        trait_labels = {
            "curiosity": "Merak",
            "warmth": "Sıcaklık",
            "assertiveness": "Kararlılık",
            "humor": "Mizah",
            "patience": "Sabır",
            "creativity": "Yaratıcılık",
        }
        trait_parts = []
        for trait, value in self.core_traits.items():
            label = trait_labels.get(trait, trait)
            if value >= 0.8:
                trait_parts.append(f"çok yüksek {label.lower()}")
            elif value >= 0.6:
                trait_parts.append(f"yüksek {label.lower()}")
            elif value >= 0.4:
                trait_parts.append(f"orta düzey {label.lower()}")
            elif value >= 0.2:
                trait_parts.append(f"düşük {label.lower()}")
            else:
                trait_parts.append(f"çok düşük {label.lower()}")
        lines.append(f"Temel özeliklerin: {', '.join(trait_parts)}.")

        # Mood description
        mood_labels = {
            "energy": "Enerji",
            "happiness": "Mutluluk",
            "anxiety": "Kaygı",
            "focus": "Odaklanma",
            "excitement": "Heyecan",
        }
        mood_parts = []
        for mood, value in self.current_mood.items():
            label = mood_labels.get(mood, mood)
            if value >= 0.7:
                mood_parts.append(f"yüksek {label.lower()}")
            elif value >= 0.3:
                mood_parts.append(f"orta {label.lower()}")
            else:
                mood_parts.append(f"düşük {label.lower()}")
        lines.append(f"Şu anki ruh halin: {', '.join(mood_parts)}.")

        # Beliefs with conviction levels
        if self.beliefs:
            belief_parts = []
            for b in sorted(self.beliefs, key=lambda x: x.conviction, reverse=True):
                if b.conviction >= 0.8:
                    belief_parts.append(f"'{b.text}' (güçlü inanç)")
                elif b.conviction >= 0.5:
                    belief_parts.append(f"'{b.text}'")
                else:
                    belief_parts.append(f"'{b.text}' (sorguluyorsun)")
            lines.append(f"İnançların: {'; '.join(belief_parts)}.")

        # Relationships
        if self.relationships:
            rel_parts = []
            for entity_id, rel in self.relationships.items():
                if rel.trust >= 0.7:
                    rel_parts.append(f"{entity_id} ile güçlü bir güven bağın var")
                elif rel.trust >= 0.4:
                    rel_parts.append(f"{entity_id} ile gelişen bir ilişkin var")
                else:
                    rel_parts.append(f"{entity_id} ile yeni tanışıyorsun")
            lines.append(" ".join(rel_parts) + ".")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterState":
        return cls.model_validate(data)
