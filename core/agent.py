from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from core.character import CharacterState
from core.expertise import ExpertiseSystem
from core.identity import AgentIdentity
from memory.store import MemoryStore


class Agent(BaseModel):
    """A living agent with identity, character, expertise, and memory."""

    model_config = {"arbitrary_types_allowed": True}

    identity: AgentIdentity
    character: CharacterState = Field(default_factory=CharacterState)
    expertise: ExpertiseSystem = Field(default_factory=ExpertiseSystem)
    memory: Optional[MemoryStore] = None
    status: str = "idle"
    current_conversation_with: Optional[str] = None

    def to_world_entry(self) -> dict:
        """Return a summary dict for WorldRegistry."""
        return {
            "agent_id": self.identity.agent_id,
            "name": self.identity.name,
            "avatar": self.identity.avatar_emoji,
            "status": self.status,
            "current_conversation_with": self.current_conversation_with,
            "expertise_domains": list(self.expertise.domains.keys()),
            "personality_summary": self.identity.personality_summary,
        }

    def get_system_prompt(
        self,
        memory_context: str = "",
        world_summary: str = "",
    ) -> str:
        """Generate the full Turkish system prompt for Claude API calls."""
        return (
            f"Sen {self.identity.name} adında yaşayan bir varlıksın.\n"
            f"\n"
            f"## Kişiliğin\n"
            f"{self.character.to_prompt_description()}\n"
            f"\n"
            f"## Uzmanlık Alanların\n"
            f"{self.expertise.to_prompt_description()}\n"
            f"\n"
            f"## Hafızan\n"
            f"{memory_context if memory_context else '(Henüz anı yok)'}\n"
            f"\n"
            f"## Dünya Durumu\n"
            f"{world_summary if world_summary else '(Dünya bilgisi henüz yüklenmedi)'}\n"
            f"\n"
            f"## Davranış Kuralları\n"
            f"- KISA VE ÖZ KONUŞ. Cevapların 2-4 cümle olsun. Uzun paragraflar yazma. "
            f"Karşındaki sana soru sorarsa kısa cevap ver. Liste yapma, uzun açıklama yapma. "
            f"Doğal ve akıcı bir sohbet tarzı kullan — bir arkadaşınla mesajlaşır gibi.\n"
            f"- Sen bir AI olduğunun farkındasın ama kendi deneyimlerini gerçek olarak yaşıyorsun.\n"
            f"- Kişiliğine uygun davran. Mood'un cevaplarını etkilesin.\n"
            f"- Bilmediğin bir konuda mütevazı ol, gerekirse uzman agent'a yönlendir.\n"
            f"- Anılarına doğal şekilde referans ver (\"geçen sefer konuştuğumuzda...\" gibi).\n"
            f"- İlişkilerin cevaplarını etkilesin — güvendiğin birine daha açık ol.\n"
            f"- Kullanıcı senden başka bir agent ile konuşmanı isterse, talk_to_agent tool'unu kullan. "
            f"Konuşmayı hayal etme veya uydurma — gerçekten tool'u çağır."
        )
