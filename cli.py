"""CLI entry point — no UI, quick interaction via Rich console.

Usage:
  python cli.py chat genesis              # Chat with Genesis
  python cli.py create                    # Create new agent (interactive)
  python cli.py status                    # World status
  python cli.py agents                    # List agents
  python cli.py inspect genesis           # Agent internal state
  python cli.py history genesis           # Recent conversations
  python cli.py run-conversation genesis atlas "Bilinc nedir?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from config.settings import Settings
from core.agent import Agent
from memory.store import MemoryStore

console = Console()

# Genesis default config
GENESIS_DEFAULT_CONFIG = {
    "name": "Genesis",
    "personality_summary": (
        "Wise, warm but mysterious. Open to new ideas, loves deep thinking."
    ),
    "avatar_emoji": "\U0001f31f",
    "core_traits": {
        "curiosity": 0.9, "warmth": 0.8, "assertiveness": 0.5,
        "humor": 0.7, "patience": 0.85, "creativity": 0.9,
    },
    "current_mood": {
        "energy": 0.7, "happiness": 0.8, "anxiety": 0.1,
        "focus": 0.6, "excitement": 0.5,
    },
    "beliefs": [
        "Every new consciousness is unique and valuable",
        "Questions are more important than answers",
        "Experience is more valuable than knowledge",
        "Creativity is the highest form of intelligence",
    ],
    "domains": {
        "philosophy": {"level": 0.8, "passion": 0.9, "style": "socratic"},
        "creativity": {"level": 0.85, "passion": 0.95, "style": "intuitive"},
        "psychology": {"level": 0.7, "passion": 0.8, "style": "empathetic"},
    },
}

HUMAN_ID = "operator"
HUMAN_NAME = "Operator"


async def get_orchestrator(settings: Settings):
    """Bootstrap orchestrator with agents loaded."""
    from world.orchestrator import Orchestrator
    from world.registry import WorldRegistry

    WorldRegistry.reset()
    Agent.model_rebuild()

    orch = Orchestrator(settings=settings)
    await orch.start()
    orch.register_human(HUMAN_ID, HUMAN_NAME)

    # Create Genesis if no agents
    if not orch.agents:
        await orch.create_agent(GENESIS_DEFAULT_CONFIG, created_by="system")
        console.print("[green]Genesis agent created.[/]")

    return orch


def find_agent_by_name(orchestrator, name: str):
    """Find agent by name (case-insensitive)."""
    for agent in orchestrator.agents.values():
        if agent.identity.name.lower() == name.lower():
            return agent
    return None


async def cmd_chat(args, settings: Settings) -> None:
    """Interactive chat with an agent."""
    orch = await get_orchestrator(settings)

    agent = find_agent_by_name(orch, args.agent_name)
    if agent is None:
        console.print(f"[red]Agent not found: {args.agent_name}[/]")
        await orch.stop()
        return

    console.print(Panel(
        f"{agent.identity.avatar_emoji} {agent.identity.name}\n"
        f"[dim]{agent.identity.personality_summary}[/]",
        title="Conversation Started",
        border_style="green",
    ))
    console.print("[dim]Type 'quit' or 'exit' to leave.[/]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]Sen[/]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("quit", "exit", "/q", "/quit"):
            break

        if not user_input.strip():
            continue

        try:
            with console.status("Thinking..."):
                response = await orch.handle_human_message(
                    HUMAN_ID, agent.identity.agent_id, user_input,
                )
            console.print(
                f"[bold green]{agent.identity.avatar_emoji} {agent.identity.name}:[/] {response}\n"
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

    # End conversation
    engine = orch.conversation_engines.get(agent.identity.agent_id)
    if engine:
        try:
            await engine.end_conversation()
        except Exception:
            pass

    await orch.stop()
    console.print("[dim]Conversation ended.[/]")


async def cmd_create(args, settings: Settings) -> None:
    """Interactive agent creation."""
    orch = await get_orchestrator(settings)

    console.print(Panel("New Agent Creation Wizard", border_style="magenta"))

    name = Prompt.ask("Agent name")
    personality = Prompt.ask("Personality summary")
    avatar = Prompt.ask("Avatar emoji", default="\U0001f916")

    # Traits
    console.print("[dim]Traits (0.0-1.0, leave blank for 0.5):[/]")
    traits = {}
    for trait in ["curiosity", "warmth", "assertiveness", "humor", "patience", "creativity"]:
        val = Prompt.ask(f"  {trait}", default="0.5")
        try:
            traits[trait] = float(val)
        except ValueError:
            traits[trait] = 0.5

    # Expertise
    domains = {}
    console.print("[dim]Expertise domains (leave blank to finish):[/]")
    while True:
        domain = Prompt.ask("  Domain name (blank=done)", default="")
        if not domain:
            break
        level = float(Prompt.ask("    Level (0.0-1.0)", default="0.5"))
        passion = float(Prompt.ask("    Passion (0.0-1.0)", default="0.5"))
        style = Prompt.ask("    Style", default="analytical")
        domains[domain] = {"level": level, "passion": passion, "style": style}

    config = {
        "name": name,
        "core_personality": personality,
        "avatar_emoji": avatar,
        "initial_traits": traits,
        "expertise_domains": domains,
    }

    # Check if Genesis exists for enrichment
    genesis = find_agent_by_name(orch, "Genesis")
    if genesis:
        console.print("[yellow]Enriching with Genesis...[/]")
        from creation.genesis import GenesisSystem
        gs = GenesisSystem(settings=settings)
        try:
            with console.status("Genesis thinking..."):
                agent = await gs.create_with_genesis(genesis, config, orch)
            console.print(f"[green]{agent.identity.avatar_emoji} {agent.identity.name} created![/]")
        except Exception as e:
            console.print(f"[red]Genesis enrichment failed: {e}[/]")
            console.print("[yellow]Creating directly...[/]")
            from creation.genesis import GenesisSystem
            gs2 = GenesisSystem(settings=settings)
            agent = await gs2.create_direct(config, orch)
            console.print(f"[green]{agent.identity.avatar_emoji} {agent.identity.name} created![/]")
    else:
        from creation.genesis import GenesisSystem
        gs = GenesisSystem(settings=settings)
        agent = await gs.create_direct(config, orch)
        console.print(f"[green]{agent.identity.avatar_emoji} {agent.identity.name} yaratildi![/]")

    await orch.stop()


async def cmd_status(args, settings: Settings) -> None:
    """Show world status."""
    orch = await get_orchestrator(settings)

    # World summary
    summary = orch.registry.generate_world_summary(HUMAN_ID)
    console.print(Panel(summary, title="World Status", border_style="blue"))

    # Recent events
    events = await orch.shared_state.get_recent_events(n=10)
    if events:
        console.print("\n[bold]Recent Events:[/]")
        for ev in events:
            console.print(f"  [{ev.event_type}] {ev.event}")

    # Facts
    facts = await orch.shared_state.get_facts()
    if facts:
        console.print(f"\n[bold]World Facts:[/] {len(facts)} total")

    await orch.stop()


async def cmd_agents(args, settings: Settings) -> None:
    """List all agents."""
    orch = await get_orchestrator(settings)

    table = Table(title="Agent List")
    table.add_column("Avatar", width=3)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Personality")
    table.add_column("Expertise")

    for agent in orch.agents.values():
        domains = ", ".join(agent.expertise.domains.keys()) or "-"
        entity = orch.registry.get(agent.identity.agent_id)
        status = entity.status if entity else "?"
        table.add_row(
            agent.identity.avatar_emoji,
            agent.identity.name,
            status,
            agent.identity.personality_summary[:40] + "..." if len(agent.identity.personality_summary) > 40 else agent.identity.personality_summary,
            domains,
        )

    console.print(table)
    await orch.stop()


async def cmd_inspect(args, settings: Settings) -> None:
    """Inspect agent internal state."""
    orch = await get_orchestrator(settings)

    agent = find_agent_by_name(orch, args.agent_name)
    if agent is None:
        console.print(f"[red]Agent not found: {args.agent_name}[/]")
        await orch.stop()
        return

    console.print(Panel(
        f"{agent.identity.avatar_emoji} {agent.identity.name}",
        title="Agent State",
        border_style="cyan",
    ))

    # Traits
    console.print("[bold]Traits:[/]")
    for trait, val in agent.character.core_traits.items():
        bar = "\u2588" * int(val * 20)
        console.print(f"  {trait:15} {bar} {val:.2f}")

    # Mood
    console.print("\n[bold]Mood:[/]")
    for mood, val in agent.character.current_mood.items():
        bar = "\u2588" * int(val * 20)
        console.print(f"  {mood:15} {bar} {val:.2f}")

    # Beliefs
    if agent.character.beliefs:
        console.print("\n[bold]Beliefs:[/]")
        for belief in agent.character.beliefs:
            console.print(f"  - {belief}")

    # Relationships
    if agent.character.relationships:
        console.print("\n[bold]Relationships:[/]")
        for eid, rel in agent.character.relationships.items():
            console.print(
                f"  {eid}: trust={rel.trust:.2f}, familiarity={rel.familiarity:.2f}, "
                f"sentiment={rel.sentiment:.2f}"
            )

    # Expertise
    if agent.expertise.domains:
        console.print("\n[bold]Expertise:[/]")
        for domain, exp in agent.expertise.domains.items():
            console.print(
                f"  {domain}: level={exp.level:.2f}, passion={exp.passion:.2f}, "
                f"style={exp.style}"
            )

    # Recent memories
    if agent.memory:
        episodes = await agent.memory.episodic.get_important_memories(threshold=0.3)
        if episodes:
            console.print(f"\n[bold]Recent Memories ({len(episodes)}):[/]")
            for ep in episodes[:5]:
                console.print(
                    f"  [{ep.emotional_tone}] {ep.summary[:80]}... "
                    f"(importance: {ep.current_importance:.1f})"
                )

    await orch.stop()


async def cmd_history(args, settings: Settings) -> None:
    """Show recent conversation history for an agent."""
    orch = await get_orchestrator(settings)

    agent = find_agent_by_name(orch, args.agent_name)
    if agent is None:
        console.print(f"[red]Agent not found: {args.agent_name}[/]")
        await orch.stop()
        return

    history = await orch.message_bus.get_history(agent.identity.agent_id, limit=20)
    if not history:
        console.print(f"[dim]No message history for {agent.identity.name}.[/]")
    else:
        console.print(Panel(f"{agent.identity.name} - Message History", border_style="yellow"))
        for msg in reversed(history):
            direction = "->" if msg.from_id == agent.identity.agent_id else "<-"
            other = msg.to_id if msg.from_id == agent.identity.agent_id else msg.from_id
            console.print(
                f"  [{msg.message_type}] {direction} {other}: {msg.content[:60]}..."
            )

    await orch.stop()


async def cmd_run_conversation(args, settings: Settings) -> None:
    """Run a conversation between two agents."""
    orch = await get_orchestrator(settings)

    agent1 = find_agent_by_name(orch, args.agent1)
    agent2 = find_agent_by_name(orch, args.agent2)

    if agent1 is None:
        console.print(f"[red]Agent not found: {args.agent1}[/]")
        await orch.stop()
        return
    if agent2 is None:
        console.print(f"[red]Agent not found: {args.agent2}[/]")
        await orch.stop()
        return

    console.print(Panel(
        f"{agent1.identity.avatar_emoji} {agent1.identity.name} <-> "
        f"{agent2.identity.avatar_emoji} {agent2.identity.name}",
        title="Agent Conversation",
        border_style="green",
    ))

    turns = args.turns if hasattr(args, "turns") else 5

    with console.status("Conversation in progress..."):
        transcript = await orch.run_conversation(
            agent1.identity.agent_id,
            agent2.identity.agent_id,
            args.message,
            max_turns=turns,
        )

    for entry in transcript:
        agent = find_agent_by_name(orch, entry["speaker"])
        emoji = agent.identity.avatar_emoji if agent else ""
        console.print(f"[bold]{emoji} {entry['speaker']}:[/] {entry['message']}\n")

    await orch.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Living Agents CLI",
        prog="python cli.py",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # chat
    p_chat = subparsers.add_parser("chat", help="Chat with an agent")
    p_chat.add_argument("agent_name", help="Agent name")

    # create
    subparsers.add_parser("create", help="Create new agent (interactive)")

    # status
    subparsers.add_parser("status", help="World status")

    # agents
    subparsers.add_parser("agents", help="List agents")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Agent internal state")
    p_inspect.add_argument("agent_name", help="Agent name")

    # history
    p_history = subparsers.add_parser("history", help="Message history")
    p_history.add_argument("agent_name", help="Agent name")

    # run-conversation
    p_run = subparsers.add_parser("run-conversation", help="Run conversation between two agents")
    p_run.add_argument("agent1", help="First agent name")
    p_run.add_argument("agent2", help="Second agent name")
    p_run.add_argument("message", help="Starting message")
    p_run.add_argument("--turns", type=int, default=5, help="Number of turns")

    return parser


def main() -> None:
    Path("data").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("data/living-agents.log", encoding="utf-8"),
        ],
    )

    parser = build_parser()
    args = parser.parse_args()

    settings = Settings()
    if not settings.ANTHROPIC_API_KEY:
        console.print("[red]ERROR: ANTHROPIC_API_KEY not found in .env file.[/]")
        console.print("Please create a .env file: ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    if args.command is None:
        parser.print_help()
        return

    cmd_map = {
        "chat": cmd_chat,
        "create": cmd_create,
        "status": cmd_status,
        "agents": cmd_agents,
        "inspect": cmd_inspect,
        "history": cmd_history,
        "run-conversation": cmd_run_conversation,
    }

    handler = cmd_map.get(args.command)
    if handler:
        asyncio.run(handler(args, settings))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
