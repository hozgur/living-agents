from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration for Living Agents framework."""

    ANTHROPIC_API_KEY: str = ""
    MODEL_NAME: str = "claude-sonnet-4-20250514"

    # Task-based model overrides (fall back to MODEL_NAME if not set)
    MODEL_CHAT: str = "claude-sonnet-4-20250514"        # Chat
    MODEL_REFLECTION: str = "claude-haiku-4-5-20251001"  # Reflection
    MODEL_AUTONOMY: str = "claude-haiku-4-5-20251001"    # Autonomy decisions
    MODEL_CREATION: str = "claude-sonnet-4-20250514"     # Agent creation
    MODEL_COMPRESSION: str = "claude-haiku-4-5-20251001" # Context compression

    # Language agents use when speaking (configurable via /language command)
    CHAT_LANGUAGE: str = "English"

    MAX_CONTEXT_TOKENS: int = 8000
    DB_PATH: str = "data/agents.db"
    CHROMA_PATH: str = "data/chroma"
    AUTONOMY_INTERVAL: int = 180
    REFLECTION_THRESHOLD: int = 3
    MEMORY_DECAY_RATE: float = 0.01
    EMBEDDING_MODEL: str = "default"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
