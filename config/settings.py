from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global configuration for Living Agents framework."""

    ANTHROPIC_API_KEY: str = ""
    MODEL_NAME: str = "claude-sonnet-4-20250514"
    MAX_CONTEXT_TOKENS: int = 8000
    DB_PATH: str = "data/agents.db"
    CHROMA_PATH: str = "data/chroma"
    AUTONOMY_INTERVAL: int = 60
    REFLECTION_THRESHOLD: int = 3
    MEMORY_DECAY_RATE: float = 0.01
    EMBEDDING_MODEL: str = "default"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
