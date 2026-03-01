"""ContextBuilder — constructs the full prompt for Claude API calls.

Combines agent identity, character, expertise, memory context, and world state
into a system prompt, and converts working memory into Claude API message format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import Agent
    from memory.working import WorkingMemory

logger = logging.getLogger(__name__)


def build_system_prompt(
    agent: Agent,
    memory_context: str = "",
    world_summary: str = "",
    language: str = "English",
) -> str:
    """Build the full system prompt for a Claude API call.

    Delegates to Agent.get_system_prompt() which already assembles
    identity, character, expertise, memory, world state, and behavior rules.
    """
    return agent.get_system_prompt(
        memory_context=memory_context,
        world_summary=world_summary,
        language=language,
    )


def build_messages(working_memory: WorkingMemory) -> list[dict[str, str]]:
    """Convert working memory into Claude API messages format.

    If there is a compressed summary, it is prepended as the first user message
    so Claude has prior conversation context.
    """
    messages: list[dict[str, str]] = []

    context = working_memory.get_context()

    # Prepend summary as context if it exists
    if context["summary"]:
        messages.append({
            "role": "user",
            "content": (
                f"[Summary of previous conversation: {context['summary']}]\n\n"
                "Please continue taking this context into account."
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "Understood, I remember our previous conversation. Let's continue.",
        })

    # Add actual conversation messages
    for msg in context["messages"]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    return messages
