"""
Context engine -- the heart of the system.

Manages:
- Session store (in-memory + JSON persistence)
- Token counting via tiktoken
- FIFO token budget eviction
- Context snapshots for rollback
- Context persistence to JSON files

This module is completely independent of the LLM.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import tiktoken

from src.config import get_logs_dir, get_token_budget_limit
from src.models import AgentContext, Message, PinnedMemory

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Token Counting
# --------------------------------------------------

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken cl100k_base.

    Args:
        text: The text to tokenize.

    Returns:
        Integer token count.
    """
    return len(_encoding.encode(text))


# --------------------------------------------------
# Token Budget Eviction
# --------------------------------------------------


def apply_token_budget(
    recent_messages: list[Message], max_tokens: int
) -> tuple[list[Message], int]:
    """
    Apply FIFO eviction to keep total tokens within budget.

    Removes the oldest messages first until the total token count
    is at or below max_tokens.

    Args:
        recent_messages: List of Message objects (will NOT be mutated).
        max_tokens: Maximum allowed token sum.

    Returns:
        Tuple of (surviving messages, total token count).
    """
    messages = list(recent_messages)  # shallow copy
    total = sum(m.tokens for m in messages)

    while total > max_tokens and messages:
        evicted = messages.pop(0)
        logger.info(
            "Evicted oldest message (role=%s, tokens=%d)", evicted.role, evicted.tokens
        )
        total -= evicted.tokens

    return messages, total


# --------------------------------------------------
# Session Store
# --------------------------------------------------

# In-memory store: session_id -> AgentContext
_session_store: dict[str, AgentContext] = {}


def _sanitize_session_id(session_id: str) -> str:
    """
    Sanitize a session ID for safe use in file paths.
    Only allows alphanumerics, hyphens, and underscores.
    Prevents path traversal attacks.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)


def get_or_create_session(session_id: str) -> AgentContext:
    """
    Retrieve an existing session or create a new one.

    On first access for a session_id, also attempts to load
    persisted context from disk (warm restart support).

    Args:
        session_id: Unique session identifier.

    Returns:
        The AgentContext for this session.
    """
    if session_id in _session_store:
        return _session_store[session_id]

    # Try loading from disk
    ctx = _load_context_from_disk(session_id)
    if ctx is not None:
        _session_store[session_id] = ctx
        logger.info("Session loaded from disk: %s", session_id)
        return ctx

    # Create fresh session
    ctx = AgentContext(session_id=session_id)
    _session_store[session_id] = ctx
    logger.info("Session created: %s", session_id)
    return ctx


def get_session(session_id: str) -> Optional[AgentContext]:
    """
    Retrieve an existing session without creating one.

    Returns:
        The AgentContext or None if session doesn't exist.
    """
    if session_id in _session_store:
        return _session_store[session_id]

    # Try loading from disk
    ctx = _load_context_from_disk(session_id)
    if ctx is not None:
        _session_store[session_id] = ctx
        return ctx

    return None


def save_session(ctx: AgentContext) -> None:
    """Save a context back into the session store."""
    _session_store[ctx.session_id] = ctx


def snapshot_context(ctx: AgentContext) -> AgentContext:
    """
    Create a deep copy of the context for rollback purposes.

    Returns:
        A completely independent copy of the AgentContext.
    """
    return ctx.model_copy(deep=True)


def restore_context(snapshot: AgentContext) -> None:
    """
    Restore a session to a previous snapshot (rollback).

    Args:
        snapshot: The previously captured snapshot.
    """
    _session_store[snapshot.session_id] = snapshot
    logger.info("Context rolled back for session: %s", snapshot.session_id)


# --------------------------------------------------
# Context Update Helpers
# --------------------------------------------------


def add_message(ctx: AgentContext, role: str, content: str) -> Message:
    """
    Add a message to the context's recent history.

    Counts tokens and updates the total.

    Args:
        ctx: The session context.
        role: "user" or "assistant".
        content: Message text.

    Returns:
        The created Message object.
    """
    tokens = count_tokens(content)
    msg = Message(role=role, content=content, tokens=tokens)
    ctx.recent.append(msg)
    ctx.total_recent_tokens += tokens
    logger.info("Added %s message (%d tokens), total=%d", role, tokens, ctx.total_recent_tokens)
    return msg


def enforce_budget(ctx: AgentContext, max_tokens: int | None = None) -> None:
    """
    Apply the token budget to the context's recent messages.

    Args:
        ctx: The session context (mutated in place).
        max_tokens: Override budget; defaults to env config.
    """
    if max_tokens is None:
        max_tokens = get_token_budget_limit()

    if ctx.total_recent_tokens <= max_tokens:
        return

    logger.info(
        "Token budget exceeded: %d > %d", ctx.total_recent_tokens, max_tokens
    )
    ctx.recent, ctx.total_recent_tokens = apply_token_budget(ctx.recent, max_tokens)


def update_pinned_ticket(ctx: AgentContext, ticket_id: str) -> None:
    """Pin a ticket ID to the context."""
    ctx.pinned.ticket_id = ticket_id
    logger.info("Ticket pinned: %s", ticket_id)


def update_topic(ctx: AgentContext, topic: str, force: bool = False) -> None:
    """
    Update the active topic.

    If the newly detected topic is 'general' and the context already has
    an active specific topic, retain the existing topic so that follow-up
    questions (e.g. "Can you give me more troubleshooting steps?") remain
    in context. To explicitly reset to general, pass force=True.
    """
    if topic == "general" and ctx.active_topic != "general" and not force:
        logger.info("Retaining active topic '%s' for continuation", ctx.active_topic)
        return

    if ctx.active_topic != topic:
        logger.info("Topic updated: %s -> %s", ctx.active_topic, topic)
        ctx.active_topic = topic


# --------------------------------------------------
# Context Persistence (JSON)
# --------------------------------------------------


def _context_filepath(session_id: str) -> Path:
    """Get the file path for a session's persisted context."""
    safe_id = _sanitize_session_id(session_id)
    logs_dir = Path(get_logs_dir())
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"context_{safe_id}.json"


def persist_context(ctx: AgentContext) -> None:
    """
    Write the context to a JSON file for observability and persistence.

    Args:
        ctx: The context to persist.
    """
    filepath = _context_filepath(ctx.session_id)
    data = ctx.model_dump()
    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Context persisted: %s -> %s", ctx.session_id, filepath)


def _load_context_from_disk(session_id: str) -> Optional[AgentContext]:
    """
    Attempt to load a persisted context from disk.

    Returns:
        AgentContext if found and valid, else None.
    """
    filepath = _context_filepath(session_id)
    if not filepath.exists():
        return None
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return AgentContext(**data)
    except Exception as exc:
        logger.warning("Failed to load context from %s: %s", filepath, exc)
        return None


# --------------------------------------------------
# Testing Helpers
# --------------------------------------------------


def clear_all_sessions(clear_disk: bool = False) -> None:
    """Clear the session store. For testing only."""
    _session_store.clear()
    if clear_disk:
        logs_dir = Path(get_logs_dir())
        if logs_dir.exists():
            for f in logs_dir.glob("context_*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass
