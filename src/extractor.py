"""
Entity and topic extractor module.

Provides deterministic extraction of:
- Ticket IDs (regex-based)
- Conversation topics (keyword/rule-based)

Designed to be independent of any LLM, fully unit-testable,
and replaceable with an LLM-based extractor in the future.
"""

from __future__ import annotations

import re
from typing import Optional


# --------------------------------------------------
# Ticket Extraction
# --------------------------------------------------

# Matches IT-XXXX where X is a digit. Uses word boundary to avoid
# matching partial strings like "ADMIT-1234".
_TICKET_PATTERN = re.compile(r"\bIT-(\d{4})\b")


def extract_ticket_id(text: str) -> Optional[str]:
    """
    Extract the *last* ticket ID from a text string.

    When multiple ticket IDs are present, the last one is returned
    (the latest mentioned ticket takes priority).

    Args:
        text: User message text.

    Returns:
        The ticket ID string (e.g. "IT-9921") or None.
    """
    matches = _TICKET_PATTERN.findall(text)
    if not matches:
        return None
    # Return the last match -- latest ticket takes priority
    return f"IT-{matches[-1]}"


# --------------------------------------------------
# Topic Detection
# --------------------------------------------------

# Controlled topic vocabulary
TOPICS = (
    "general",
    "ticket_inquiry",
    "wifi_issue",
    "printing_issue",
    "network_issue",
    "password_issue",
    "software_issue",
)

# Keyword -> topic mapping. Order matters: first match wins.
# More specific patterns come before broader ones.
_TOPIC_RULES: list[tuple[list[str], str]] = [
    # Ticket inquiry (check before others because "ticket" is specific)
    (
        ["ticket", "IT-", "ticket number", "ticket id", "ticket status"],
        "ticket_inquiry",
    ),
    # WiFi issues
    (
        ["wifi", "wi-fi", "wireless", "wlan", "connected without internet",
         "no internet", "hotspot", "access point"],
        "wifi_issue",
    ),
    # Printing issues
    (
        ["print", "printer", "printing", "paper jam", "toner", "scanner"],
        "printing_issue",
    ),
    # Network issues (broader than WiFi)
    (
        ["network", "ethernet", "vpn", "dns", "firewall", "proxy",
         "connection refused", "timeout", "latency"],
        "network_issue",
    ),
    # Password issues
    (
        ["password", "reset password", "forgot password", "locked out",
         "login", "sign in", "credentials", "two-factor", "2fa", "mfa"],
        "password_issue",
    ),
    # Software issues
    (
        ["software", "install", "update", "app", "application", "license",
         "crash", "error message", "blue screen", "bsod", "driver",
         "microsoft", "office", "excel", "word", "zoom"],
        "software_issue",
    ),
]


def detect_topic(text: str) -> str:
    """
    Detect the conversation topic from a user message.

    Uses rule-based keyword matching against a controlled vocabulary.
    Falls back to "general" when no specific topic is detected.

    Args:
        text: User message text.

    Returns:
        Topic string from the TOPICS vocabulary.
    """
    lower = text.lower()
    for keywords, topic in _TOPIC_RULES:
        for kw in keywords:
            if kw.lower() in lower:
                return topic
    return "general"
