"""Main entry point — launches the Terminal UI.

1. Load config
2. Create Orchestrator
3. Load saved agents (or create Genesis)
4. Register human operator
5. Start Terminal UI
6. Graceful shutdown on Ctrl+C
"""

import logging
import sys

from config.settings import Settings
from ui.terminal_app import LivingAgentsApp


def setup_logging() -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("data/living-agents.log", encoding="utf-8"),
        ],
    )
    # Claude API calls at DEBUG level
    logging.getLogger("conversation.engine").setLevel(logging.DEBUG)
    logging.getLogger("conversation.reflection").setLevel(logging.DEBUG)


def main() -> None:
    """Launch the Living Agents TUI."""
    # Ensure data directory exists
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)

    setup_logging()

    settings = Settings()
    if not settings.ANTHROPIC_API_KEY:
        print("HATA: ANTHROPIC_API_KEY .env dosyasinda bulunamadi.")
        print("Lutfen .env dosyasi olusturun: ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    app = LivingAgentsApp(settings=settings)
    app.run()


if __name__ == "__main__":
    main()
