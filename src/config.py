"""
Configuration module.

Loads environment variables and provides typed configuration
for the entire application. All config is centralized here.
"""

import os
from dotenv import load_dotenv

# Load .env file if present (no-op in production with real env vars)
load_dotenv()


def get_openai_api_key() -> str:
    """Return the OpenAI API key. Never log or expose this value."""
    return os.getenv("OPENAI_API_KEY", "")


def get_openai_model() -> str:
    """Return the OpenAI model name."""
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_token_budget_limit() -> int:
    """Return the token budget limit for recent messages."""
    raw = os.getenv("TOKEN_BUDGET_LIMIT", "4000")
    try:
        return int(raw)
    except ValueError:
        return 4000


def get_llm_provider() -> str:
    """Return the LLM provider: 'openai' or 'mock'."""
    return os.getenv("LLM_PROVIDER", "mock").lower()


def get_logs_dir() -> str:
    """Return the path to the logs directory."""
    return os.getenv("LOGS_DIR", "logs")
