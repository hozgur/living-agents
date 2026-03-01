from typing import Any, Optional

from pydantic import BaseModel, Field


class DomainExpertise(BaseModel):
    """Expertise in a specific domain."""

    level: float = Field(default=0.0, ge=0.0, le=1.0)
    passion: float = Field(default=0.5, ge=0.0, le=1.0)
    style: str = "analytical"


class ExpertiseSystem(BaseModel):
    """Manages an agent's knowledge domains and learning."""

    domains: dict[str, DomainExpertise] = Field(default_factory=dict)
    learning_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    teaching_style: str = "step_by_step"

    def get_confidence(self, domain: str) -> float:
        """Return confidence level for a domain (0.0 if unknown)."""
        if domain not in self.domains:
            return 0.0
        expertise = self.domains[domain]
        return expertise.level

    def learn(self, domain: str, amount: float) -> None:
        """Increase knowledge in a domain, weighted by learning_rate."""
        if domain not in self.domains:
            self.domains[domain] = DomainExpertise()
        expertise = self.domains[domain]
        effective_amount = amount * self.learning_rate
        expertise.level = min(1.0, expertise.level + effective_amount)

    def get_expert_for(self, domain: str, world_registry: Optional[Any] = None) -> Optional[str]:
        """Find an agent with better expertise in this domain.

        Skeleton — requires WorldRegistry from Phase 4.
        """
        if world_registry is None:
            return None
        # Phase 4: query world_registry for agents with higher expertise in domain
        return None

    def to_prompt_description(self, language: str = "English") -> str:
        """Generate natural language description for system prompt."""
        if not self.domains:
            return "You don't have a specific area of expertise yet, but you're open to learning."

        lines = []
        style_labels = {
            "socratic": "Socratic questioning",
            "analytical": "analytical approach",
            "creative": "creative thinking",
            "intuitive": "intuitive grasp",
            "empathetic": "empathetic understanding",
            "cautious_learner": "cautious learning",
            "step_by_step": "step-by-step explanation",
            "metaphor_heavy": "metaphor-heavy narration",
            "example_driven": "example-driven teaching",
        }

        for domain, expertise in self.domains.items():
            level_desc = (
                "expert" if expertise.level >= 0.8
                else "advanced" if expertise.level >= 0.6
                else "intermediate" if expertise.level >= 0.4
                else "beginner" if expertise.level >= 0.2
                else "novice"
            )
            passion_desc = (
                "passionately devoted to" if expertise.passion >= 0.8
                else "interested in" if expertise.passion >= 0.5
                else "familiar with"
            )
            style_desc = style_labels.get(expertise.style, expertise.style)
            lines.append(
                f"- {domain.capitalize()}: {level_desc} level, a field you are {passion_desc}. "
                f"Your approach: {style_desc}."
            )

        teaching_desc = style_labels.get(self.teaching_style, self.teaching_style)
        lines.append(f"\nTeaching style: {teaching_desc}.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertiseSystem":
        return cls.model_validate(data)
