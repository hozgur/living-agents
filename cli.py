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
        "Bilge, s\u0131cak ama gizemli. Yeni fikirlere a\u00e7\u0131k, derin d\u00fc\u015f\u00fcnmeyi sever."
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
        "Her yeni bilin\u00e7 benzersiz ve de\u011ferli",
        "Sorular cevaplardan daha \u00f6nemli",
        "Deneyim bilgiden daha de\u011ferli",
        "Yarat\u0131c\u0131l\u0131k en y\u00fcksek zeka bi\u00e7imi",
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
        console.print("[green]Genesis agent olusturuldu.[/]")

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
        console.print(f"[red]Agent bulunamadi: {args.agent_name}[/]")
        await orch.stop()
        return

    console.print(Panel(
        f"{agent.identity.avatar_emoji} {agent.identity.name}\n"
        f"[dim]{agent.identity.personality_summary}[/]",
        title="Konusma Basladi",
        border_style="green",
    ))
    console.print("[dim]Cikmak icin 'quit' veya 'exit' yazin.[/]\n")

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
            with console.status("Dusunuyor..."):
                response = await orch.handle_human_message(
                    HUMAN_ID, agent.identity.agent_id, user_input,
                )
            console.print(
                f"[bold green]{agent.identity.avatar_emoji} {agent.identity.name}:[/] {response}\n"
            )
        except Exception as e:
            console.print(f"[red]Hata: {e}[/]")

    # End conversation
    engine = orch.conversation_engines.get(agent.identity.agent_id)
    if engine:
        try:
            await engine.end_conversation()
        except Exception:
            pass

    await orch.stop()
    console.print("[dim]Konusma sonlandi.[/]")


async def cmd_create(args, settings: Settings) -> None:
    """Interactive agent creation."""
    orch = await get_orchestrator(settings)

    console.print(Panel("Yeni Agent Yaratma Sihirbazi", border_style="magenta"))

    name = Prompt.ask("Agent adi")
    personality = Prompt.ask("Kisilik ozeti")
    avatar = Prompt.ask("Avatar emoji", default="\U0001f916")

    # Traits
    console.print("[dim]Ozellikler (0.0-1.0, bos birakirsaniz 0.5):[/]")
    traits = {}
    for trait in ["curiosity", "warmth", "assertiveness", "humor", "patience", "creativity"]:
        val = Prompt.ask(f"  {trait}", default="0.5")
        try:
            traits[trait] = float(val)
        except ValueError:
            traits[trait] = 0.5

    # Expertise
    domains = {}
    console.print("[dim]Uzmanlik alanlari (bos birakirsaniz bitir):[/]")
    while True:
        domain = Prompt.ask("  Alan adi (bos=bitir)", default="")
        if not domain:
            break
        level = float(Prompt.ask("    Seviye (0.0-1.0)", default="0.5"))
        passion = float(Prompt.ask("    Tutku (0.0-1.0)", default="0.5"))
        style = Prompt.ask("    Stil", default="analytical")
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
        console.print("[yellow]Genesis ile zenginlestiriliyor...[/]")
        from creation.genesis import GenesisSystem
        gs = GenesisSystem(settings=settings)
        try:
            with console.status("Genesis dusunuyor..."):
                agent = await gs.create_with_genesis(genesis, config, orch)
            console.print(f"[green]{agent.identity.avatar_emoji} {agent.identity.name} yaratildi![/]")
        except Exception as e:
            console.print(f"[red]Genesis zenginlestirme basarisiz: {e}[/]")
            console.print("[yellow]Dogrudan yaratiliyor...[/]")
            from creation.genesis import GenesisSystem
            gs2 = GenesisSystem(settings=settings)
            agent = await gs2.create_direct(config, orch)
            console.print(f"[green]{agent.identity.avatar_emoji} {agent.identity.name} yaratildi![/]")
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
    console.print(Panel(summary, title="Dunya Durumu", border_style="blue"))

    # Recent events
    events = await orch.shared_state.get_recent_events(n=10)
    if events:
        console.print("\n[bold]Son Olaylar:[/]")
        for ev in events:
            console.print(f"  [{ev.event_type}] {ev.event}")

    # Facts
    facts = await orch.shared_state.get_facts()
    if facts:
        console.print(f"\n[bold]Dunya Gercekleri:[/] {len(facts)} adet")

    await orch.stop()


async def cmd_agents(args, settings: Settings) -> None:
    """List all agents."""
    orch = await get_orchestrator(settings)

    table = Table(title="Agent Listesi")
    table.add_column("Avatar", width=3)
    table.add_column("Isim", style="bold")
    table.add_column("Durum")
    table.add_column("Kisilik")
    table.add_column("Uzmanlik")

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
        console.print(f"[red]Agent bulunamadi: {args.agent_name}[/]")
        await orch.stop()
        return

    console.print(Panel(
        f"{agent.identity.avatar_emoji} {agent.identity.name}",
        title="Agent Durumu",
        border_style="cyan",
    ))

    # Traits
    console.print("[bold]Ozellikler:[/]")
    for trait, val in agent.character.core_traits.items():
        bar = "\u2588" * int(val * 20)
        console.print(f"  {trait:15} {bar} {val:.2f}")

    # Mood
    console.print("\n[bold]Ruh Hali:[/]")
    for mood, val in agent.character.current_mood.items():
        bar = "\u2588" * int(val * 20)
        console.print(f"  {mood:15} {bar} {val:.2f}")

    # Beliefs
    if agent.character.beliefs:
        console.print("\n[bold]Inanclar:[/]")
        for belief in agent.character.beliefs:
            console.print(f"  - {belief}")

    # Relationships
    if agent.character.relationships:
        console.print("\n[bold]Iliskiler:[/]")
        for eid, rel in agent.character.relationships.items():
            console.print(
                f"  {eid}: guven={rel.trust:.2f}, asinalik={rel.familiarity:.2f}, "
                f"duygu={rel.sentiment:.2f}"
            )

    # Expertise
    if agent.expertise.domains:
        console.print("\n[bold]Uzmanlik:[/]")
        for domain, exp in agent.expertise.domains.items():
            console.print(
                f"  {domain}: seviye={exp.level:.2f}, tutku={exp.passion:.2f}, "
                f"stil={exp.style}"
            )

    # Recent memories
    if agent.memory:
        episodes = await agent.memory.episodic.get_important_memories(threshold=0.3)
        if episodes:
            console.print(f"\n[bold]Son Anilar ({len(episodes)}):[/]")
            for ep in episodes[:5]:
                console.print(
                    f"  [{ep.emotional_tone}] {ep.summary[:80]}... "
                    f"(onem: {ep.current_importance:.1f})"
                )

    await orch.stop()


async def cmd_history(args, settings: Settings) -> None:
    """Show recent conversation history for an agent."""
    orch = await get_orchestrator(settings)

    agent = find_agent_by_name(orch, args.agent_name)
    if agent is None:
        console.print(f"[red]Agent bulunamadi: {args.agent_name}[/]")
        await orch.stop()
        return

    history = await orch.message_bus.get_history(agent.identity.agent_id, limit=20)
    if not history:
        console.print(f"[dim]{agent.identity.name} icin mesaj gecmisi yok.[/]")
    else:
        console.print(Panel(f"{agent.identity.name} - Mesaj Gecmisi", border_style="yellow"))
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
        console.print(f"[red]Agent bulunamadi: {args.agent1}[/]")
        await orch.stop()
        return
    if agent2 is None:
        console.print(f"[red]Agent bulunamadi: {args.agent2}[/]")
        await orch.stop()
        return

    console.print(Panel(
        f"{agent1.identity.avatar_emoji} {agent1.identity.name} <-> "
        f"{agent2.identity.avatar_emoji} {agent2.identity.name}",
        title="Agent Konusmasi",
        border_style="green",
    ))

    turns = args.turns if hasattr(args, "turns") else 5

    with console.status("Konusma devam ediyor..."):
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
    subparsers = parser.add_subparsers(dest="command", help="Komutlar")

    # chat
    p_chat = subparsers.add_parser("chat", help="Agent ile konus")
    p_chat.add_argument("agent_name", help="Agent adi")

    # create
    subparsers.add_parser("create", help="Yeni agent yarat (interaktif)")

    # status
    subparsers.add_parser("status", help="Dunya durumu")

    # agents
    subparsers.add_parser("agents", help="Agent listesi")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Agent ic durumu")
    p_inspect.add_argument("agent_name", help="Agent adi")

    # history
    p_history = subparsers.add_parser("history", help="Mesaj gecmisi")
    p_history.add_argument("agent_name", help="Agent adi")

    # run-conversation
    p_run = subparsers.add_parser("run-conversation", help="Iki agent konusturt")
    p_run.add_argument("agent1", help="Birinci agent adi")
    p_run.add_argument("agent2", help="Ikinci agent adi")
    p_run.add_argument("message", help="Baslangic mesaji")
    p_run.add_argument("--turns", type=int, default=5, help="Tur sayisi")

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
        console.print("[red]HATA: ANTHROPIC_API_KEY .env dosyasinda bulunamadi.[/]")
        console.print("Lutfen .env dosyasi olusturun: ANTHROPIC_API_KEY=sk-...")
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
