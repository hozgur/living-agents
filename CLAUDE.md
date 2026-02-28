# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Living Agents is a multi-agent framework where AI agents have evolving personalities, layered memory, and autonomous behavior. Agents speak Turkish by default. The full specification lives in `living-agent-spec.md`.

**Stack:** Python 3.11+, Claude API (anthropic SDK, model: claude-sonnet-4-20250514), SQLite, ChromaDB, asyncio, Textual (TUI), Rich, Pydantic

**No external agent frameworks** (LangChain, LangGraph, etc.) — everything is built from scratch.

## Build & Run

```bash
# Install dependencies
pip install -e .

# Run with Terminal UI
python main.py

# CLI (no UI)
python cli.py chat genesis
python cli.py create
python cli.py status
python cli.py agents
python cli.py run-conversation genesis atlas "Bilinç nedir?"
```

Requires `.env` with `ANTHROPIC_API_KEY`. First run auto-creates `data/` directory, SQLite DB, Genesis Agent, and registers human as "Operator".

## Architecture

### Implementation Phases (build in order)

1. **Core** (`core/`) — Agent identity, character traits (0.0-1.0 evolving), expertise system
2. **Memory** (`memory/`) — Three-layer memory: episodic (SQLite+ChromaDB), semantic (triple-store), working (token-aware context)
3. **Conversation** (`conversation/`) — Context builder, async Claude API engine, post-conversation reflection producing structured JSON
4. **World** (`world/`) — Registry (singleton), asyncio.Queue message bus, shared state, orchestrator with autonomy loops
5. **Creation** (`creation/`) — Genesis system for agent creation with personality enrichment
6. **UI** (`ui/`) — Textual app with God Mode (omniscient) and Participant Mode (single-agent perspective)
7. **Entry Points** — `main.py` (TUI) and `cli.py` (argparse + Rich)

### Key Patterns

- **All I/O is async** — Claude API calls, SQLite, message bus all use `await`
- **Reflection-based learning** — After every N messages (REFLECTION_THRESHOLD=5), agents self-reflect via Claude, producing JSON with episode summaries, mood/trait deltas, relationship updates, and learned facts
- **Character evolution** — Traits change slowly (max ±0.02 per interaction), mood changes fast (0.0-1.0 continuous)
- **Memory decay** — Episodic importance decays at MEMORY_DECAY_RATE (0.01/day); emotionally intense memories decay slower
- **Working memory compression** — At 80% token capacity, older messages are summarized via Claude and replaced with the summary
- **Autonomy loop** — Every AUTONOMY_INTERVAL (300s), agents decide autonomously: talk to someone, reflect, or idle
- **WorldRegistry is singleton** — Global entity tracking for all agents and humans

### Data Flow

Human/Agent message → `Orchestrator.handle_*_message()` → `ConversationEngine.chat()` → builds context (identity + character + expertise + memory recall + world summary) → Claude API → response stored in working memory → reflection triggered at threshold → `ReflectionEngine` parses JSON → updates character, relationships, episodic/semantic memory

### Database

SQLite at `data/agents.db` with tables: `agents`, `episodes`, `knowledge_facts`, `messages`, `world_events`, `world_facts`, `conversations`. Full schema in spec file.

ChromaDB at `data/chroma` for embedding-based semantic search on episodic memories (per-agent collections).

## Implementation Guidelines

- Exponential backoff for Claude API retries (rate limits, timeouts)
- Always try/except when parsing reflection JSON from Claude — define fallback behavior for malformed responses
- Graceful shutdown on Ctrl+C must save all agent states to SQLite
- Use Python `logging` module with structured logging; DEBUG level logs all Claude API calls
- Each module must be independently testable with Claude API mockable
- System prompts and agent default language are Turkish
