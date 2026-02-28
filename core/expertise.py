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

    def to_prompt_description(self) -> str:
        """Generate Turkish natural language description for system prompt."""
        if not self.domains:
            return "Henüz belirli bir uzmanlık alanın yok, öğrenmeye açıksın."

        lines = []
        style_labels = {
            "socratic": "sokratik sorgulama",
            "analytical": "analitik yaklaşım",
            "creative": "yaratıcı düşünme",
            "intuitive": "sezgisel kavrayış",
            "empathetic": "empatik anlayış",
            "cautious_learner": "temkinli öğrenme",
            "step_by_step": "adım adım açıklama",
            "metaphor_heavy": "metafor ağırlıklı anlatım",
            "example_driven": "örnek odaklı öğretim",
        }

        for domain, expertise in self.domains.items():
            level_desc = (
                "uzman" if expertise.level >= 0.8
                else "ileri düzey" if expertise.level >= 0.6
                else "orta düzey" if expertise.level >= 0.4
                else "başlangıç düzeyi" if expertise.level >= 0.2
                else "yeni başlayan"
            )
            passion_desc = (
                "tutkuyla bağlı olduğun" if expertise.passion >= 0.8
                else "ilgilendiğin" if expertise.passion >= 0.5
                else "tanıdığın"
            )
            style_desc = style_labels.get(expertise.style, expertise.style)
            lines.append(
                f"- {domain.capitalize()}: {level_desc} seviyede, {passion_desc} bir alan. "
                f"Yaklaşımın: {style_desc}."
            )

        teaching_desc = style_labels.get(self.teaching_style, self.teaching_style)
        lines.append(f"\nÖğretme tarzın: {teaching_desc}.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertiseSystem":
        return cls.model_validate(data)
