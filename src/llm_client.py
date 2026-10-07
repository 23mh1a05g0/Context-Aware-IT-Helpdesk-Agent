"""
LLM client module.

Provides a clean abstraction over LLM providers:
  - OpenAI (production)
  - Mock (testing / demo)

The mock provider inspects the constructed prompt to produce
contextually-aware deterministic responses, proving that
pinned memory is actually injected into the prompt.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from src.config import get_llm_provider, get_openai_api_key, get_openai_model

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts in OpenAI chat format.

        Returns:
            The assistant's response text.

        Raises:
            LLMError: If the LLM call fails.
        """
        ...


class LLMError(Exception):
    """Raised when an LLM API call fails."""


# --------------------------------------------------
# OpenAI Provider
# --------------------------------------------------


class OpenAIClient(LLMClient):
    """Production LLM client using the OpenAI API."""

    def __init__(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package is required for OpenAI provider") from exc

        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = get_openai_model()
        logger.info("OpenAI client initialized with model=%s", self._model)

    async def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=1024,
            )
            content = response.choices[0].message.content or ""
            logger.info("OpenAI response received (%d chars)", len(content))
            return content
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise LLMError(f"OpenAI API call failed: {exc}") from exc


# --------------------------------------------------
# Mock Provider
# --------------------------------------------------

# Ticket pattern used to find IDs in prompt
_TICKET_RE = re.compile(r"\bIT-(\d{4})\b")


class MockLLMClient(LLMClient):
    """
    Deterministic mock LLM for testing.

    Inspects the constructed prompt (system message + recent messages)
    to produce contextually-aware responses. This proves that pinned
    memory and topic tracking are actually flowing through to the LLM.
    """

    async def generate(self, messages: list[dict[str, str]]) -> str:
        # Combine all message content for inspection
        full_prompt = "\n".join(m["content"] for m in messages)

        # Find the last user message
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        last_user_lower = last_user.lower()

        # Extract ticket IDs present in the full prompt (from pinned facts)
        ticket_ids = _TICKET_RE.findall(full_prompt)
        ticket_id = f"IT-{ticket_ids[-1]}" if ticket_ids else None

        # Determine topic from system message
        system_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

        # -- Response logic based on prompt content --

        # User asks about ticket number / wants confirmation
        if ticket_id and any(
            kw in last_user_lower
            for kw in ["ticket number", "ticket id", "confirm", "what was", "go back", "original ticket"]
        ):
            return (
                f"Your ticket number is {ticket_id}. I still have it saved from "
                f"our conversation. However, I don't have access to the university's "
                f"ticketing system to retrieve its live status."
            )

        # User asks about ticket status
        if ticket_id and any(
            kw in last_user_lower for kw in ["status", "update"]
        ):
            return (
                f"I still have your ticket ID {ticket_id} from our conversation, "
                f"but I don't have access to the university's ticketing system "
                f"to retrieve its live status. You may want to check the IT portal "
                f"or contact the helpdesk directly for a status update."
            )

        # WiFi troubleshooting
        if "wifi_issue" in system_msg or any(
            kw in last_user_lower for kw in ["wifi", "wireless", "wi-fi"]
        ):
            # Check if Mac is mentioned anywhere in the prompt
            if "mac" in full_prompt.lower():
                return (
                    "Let's troubleshoot your WiFi connection on Mac. Here are the steps:\n\n"
                    "1. Click the WiFi icon in the menu bar and toggle WiFi off, then back on.\n"
                    "2. Go to System Settings -> Network -> WiFi -> Details on your network.\n"
                    "3. Click 'Forget This Network', then reconnect.\n"
                    "4. Open Terminal and run: sudo dscacheutil -flushcache\n"
                    "5. Reset network settings: Go to System Settings -> Network -> "
                    "click '...' -> Set Service Order -> Reset.\n"
                    "6. Try connecting to a different network to isolate the issue.\n"
                    "7. If the problem persists, check if other devices can connect "
                    "to the same network."
                )
            return (
                "Let's troubleshoot your WiFi connection. "
                "Please try the following steps:\n\n"
                "1. Toggle your WiFi off and on.\n"
                "2. Forget the network and reconnect.\n"
                "3. Restart your device.\n"
                "4. Check if other devices can connect to the same network.\n"
                "5. If the issue persists, contact the IT helpdesk."
            )

        # Ticket mentioned
        if ticket_id and "ticket" in last_user_lower:
            return (
                f"I've noted your ticket ID {ticket_id}. I don't have access to "
                f"the university's ticketing system to look up its details, but "
                f"I'll keep this ticket ID associated with our conversation. "
                f"How can I help you today?"
            )

        # Printing issues
        if "printing_issue" in system_msg or "print" in last_user_lower:
            return (
                "Let's troubleshoot your printing issue. "
                "Please check the following:\n\n"
                "1. Ensure the printer is powered on and connected.\n"
                "2. Check for paper jams.\n"
                "3. Verify the correct printer is selected.\n"
                "4. Try removing and re-adding the printer.\n"
                "5. Restart the print spooler service."
            )

        # Password issues
        if "password_issue" in system_msg or "password" in last_user_lower:
            return (
                "For password-related issues:\n\n"
                "1. Go to the university's password reset portal.\n"
                "2. Click 'Forgot Password'.\n"
                "3. Enter your university email.\n"
                "4. Follow the instructions sent to your recovery email.\n"
                "5. If you're still locked out, visit the IT helpdesk in person "
                "with your student/faculty ID."
            )

        # Default greeting / general
        return (
            "Hello! I'm the university IT helpdesk assistant. "
            "I can help you with WiFi issues, printing problems, "
            "password resets, software installation, network configuration, "
            "and ticket inquiries. How can I assist you today?"
        )


# --------------------------------------------------
# Factory
# --------------------------------------------------

# Singleton instance
_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """
    Get or create the LLM client based on the configured provider.

    Returns:
        An LLMClient instance.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    provider = get_llm_provider()
    if provider == "openai":
        _client_instance = OpenAIClient()
    elif provider == "mock":
        _client_instance = MockLLMClient()
        logger.info("Using mock LLM client")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    return _client_instance


def reset_llm_client() -> None:
    """Reset the LLM client singleton. For testing only."""
    global _client_instance
    _client_instance = None
