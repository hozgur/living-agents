from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class RelationshipState(BaseModel):
    """Tracks an agent's relationship with another entity."""

    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    familiarity: float = Field(default=0.0, ge=0.0, le=1.0)
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    shared_experience_count: int = 0
    last_interaction: Optional[datetime] = None
    notes: list[str] = Field(default_factory=list)


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
    beliefs: list[str] = Field(default_factory=list)
    relationships: dict[str, RelationshipState] = Field(default_factory=dict)

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

    def add_belief(self, belief: str) -> None:
        if belief not in self.beliefs:
            self.beliefs.append(belief)

    def remove_belief(self, belief: str) -> None:
        if belief in self.beliefs:
            self.beliefs.remove(belief)

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

        # Beliefs
        if self.beliefs:
            beliefs_str = "; ".join(self.beliefs)
            lines.append(f"İnançların: {beliefs_str}.")

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
