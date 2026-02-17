"""Configuration via environment variables."""
import os
from pathlib import Path

# LLM backend (any OpenAI-compatible API)
LLM_URL = os.getenv("LLM_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-change-me")
LLM_MODEL = os.getenv("LLM_MODEL", "default")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Client API keys (comma-separated)
CLIENT_API_KEYS = set(
    k.strip() for k in os.getenv("CLIENT_API_KEYS", "change-me-in-production").split(",") if k.strip()
)

# CORS
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
]

# Limits
MAX_TOKENS_CONTEXT = int(os.getenv("MAX_TOKENS_CONTEXT", "28000"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
RATE_LIMIT_PER_HOUR_SESSION = int(os.getenv("RATE_LIMIT_PER_HOUR_SESSION", "100"))
MAX_MESSAGES_PER_SESSION = 50
SESSION_TTL = 86400  # 24h

# System prompt
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_system_prompt(client: str = "default") -> str:
    path = PROMPTS_DIR / f"{client}.txt"
    if not path.exists():
        path = PROMPTS_DIR / "default.txt"
    return path.read_text().strip()

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8083"))
