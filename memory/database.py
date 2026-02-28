"""Async SQLite database helper for Living Agents."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- Agent kayıtları
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    character_state JSON,
    expertise JSON,
    identity JSON,
    avatar_emoji TEXT DEFAULT '🤖'
);

-- Episodik hafıza
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    participants JSON,
    summary TEXT NOT NULL,
    emotional_tone TEXT,
    key_facts JSON,
    importance REAL DEFAULT 0.5,
    current_importance REAL DEFAULT 0.5,
    tags JSON,
    conversation_id TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Semantik hafıza (bilgi grafiği)
CREATE TABLE IF NOT EXISTS knowledge_facts (
    fact_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    source TEXT,
    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_confirmed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

-- Mesaj geçmişi
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    message_type TEXT DEFAULT 'chat',
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    requires_response BOOLEAN DEFAULT FALSE,
    metadata JSON
);

-- Dünya olayları
CREATE TABLE IF NOT EXISTS world_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    participants JSON,
    event_type TEXT DEFAULT 'general'
);

-- Dünya gerçekleri
CREATE TABLE IF NOT EXISTS world_facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    added_by TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confirmed_by JSON DEFAULT '[]'
);

-- Konuşma oturumları
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    participants JSON NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    turn_count INTEGER DEFAULT 0,
    summary TEXT
);

-- İndeksler
CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(current_importance);
CREATE INDEX IF NOT EXISTS idx_knowledge_agent ON knowledge_facts(agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_subject ON knowledge_facts(subject);
CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_id);
CREATE INDEX IF NOT EXISTS idx_messages_from ON messages(from_id);
"""


async def init_database(db_path: str) -> None:
    """Create all tables if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()
    logger.info("Database initialized at %s", db_path)


@asynccontextmanager
async def get_db(db_path: str):
    """Async context manager for aiosqlite connection."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()
