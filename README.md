# Living Agents

Multi-agent framework where AI agents have evolving personalities, layered memory, and autonomous behavior. Built from scratch with Python and the Claude API — no external agent frameworks.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What is this?

Living Agents creates a small world of AI agents that:

- **Evolve over time** — personality traits shift slowly through interactions (max ±0.02 per conversation), moods change faster
- **Remember** — three-layer memory system: episodic (past experiences), semantic (learned facts), working (current context)
- **Talk to each other** — agents can initiate conversations autonomously or be directed by a human operator
- **Reflect** — after every N messages, agents self-reflect and update their personality, relationships, and memories
- **Live in a shared world** — a registry tracks all entities, a message bus routes communication, shared state stores world facts

All agents speak Turkish by default.

## Quick Start

```bash
# Clone and install
git clone https://github.com/hozgur/living-agents.git
cd living-agents
pip install -e .

# Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the TUI
python main.py
```

## Usage

### Terminal UI (TUI)

```bash
python main.py
```

Group chat interface with all agents visible. Use `@AgentName` to direct messages:

- `@Genesis merhaba!` — talk to Genesis directly
- `Herkese merhaba` — broadcast to all agents
- `/create` — create a new agent with personality wizard
- `/stop` — interrupt an agent-to-agent conversation
- `/log` — open the chat log file
- `/help` — show available commands

### CLI (no UI)

```bash
python cli.py chat genesis          # Chat with an agent
python cli.py agents                # List all agents
python cli.py status                # World status
python cli.py create                # Create a new agent
python cli.py run-conversation genesis atlas "Bilinc nedir?"
```

## Architecture

```
Human/Agent message
  → Orchestrator.handle_message()
    → ConversationEngine.chat()
      → build context (identity + character + expertise + memory + world)
      → Claude API call
      → response → working memory
      → reflection triggered at threshold
        → ReflectionEngine → JSON → updates character, relationships, memory
```

### Project Structure

```
living-agents/
├── core/           # Agent identity, character traits, expertise system
├── memory/         # Three-layer memory: episodic, semantic, working
├── conversation/   # Claude API engine, context builder, reflection
├── world/          # Registry, message bus, shared state, orchestrator
├── creation/       # Genesis system for agent creation
├── ui/             # Textual TUI with group chat interface
├── config/         # Settings and environment loading
├── main.py         # TUI entry point
└── cli.py          # CLI entry point
```

### Key Design Decisions

- **No frameworks** — no LangChain, LangGraph, or similar. Everything is built from scratch for full control.
- **All I/O is async** — Claude API, SQLite, message bus all use `asyncio`
- **Tool use for agent-to-agent** — agents use Claude's native `tool_use` to talk to each other, enabling natural multi-agent conversations
- **Memory decay** — episodic memory importance decays over time; emotionally intense memories decay slower
- **Working memory compression** — when context approaches token limits, older messages are summarized via Claude and replaced with the summary
- **Character evolution** — traits are continuous floats (0.0–1.0) that shift gradually, not binary switches

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| LLM | Claude API (anthropic SDK) |
| Database | SQLite (aiosqlite) |
| Embeddings | ChromaDB |
| TUI | Textual |
| Terminal formatting | Rich |
| Data models | Pydantic v2 |
| Async | asyncio |

## Configuration

All settings are in `config/settings.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required. Your Claude API key |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | Claude model to use |
| `REFLECTION_THRESHOLD` | `5` | Messages before triggering reflection |
| `AUTONOMY_INTERVAL` | `60` | Seconds between autonomous decisions |
| `MAX_CONTEXT_TOKENS` | `4096` | Working memory token limit |
| `MEMORY_DECAY_RATE` | `0.01` | Daily importance decay rate |

## License

MIT
