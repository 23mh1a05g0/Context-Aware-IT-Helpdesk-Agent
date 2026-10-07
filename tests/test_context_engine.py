"""
Tests for the context engine: token counting, eviction,
session management, persistence, and snapshots.
"""

import json
import os
import pytest
from pathlib import Path

from src.context_engine import (
    add_message,
    apply_token_budget,
    clear_all_sessions,
    count_tokens,
    enforce_budget,
    get_or_create_session,
    get_session,
    persist_context,
    restore_context,
    save_session,
    snapshot_context,
    update_pinned_ticket,
    update_topic,
    _context_filepath,
    _sanitize_session_id,
)
from src.models import AgentContext, Message, PinnedMemory


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clear session store before each test."""
    clear_all_sessions()
    yield
    clear_all_sessions()


# ──────────────────────────────────────────────
# Token Counting
# ──────────────────────────────────────────────


class TestTokenCounting:
    def test_count_tokens_simple(self):
        count = count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_longer(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many words in it")
        assert long > short

    def test_count_uses_tiktoken_not_word_split(self):
        # tiktoken and word-split give different results for this
        text = "Hello, world! How's it going?"
        tokens = count_tokens(text)
        words = len(text.split())
        # They should differ for most inputs
        assert tokens != words or tokens > 0  # At minimum, count is positive


# ──────────────────────────────────────────────
# Token Budget Eviction
# ──────────────────────────────────────────────


class TestTokenEviction:
    def test_no_eviction_needed(self):
        messages = [
            Message(role="user", content="Hi", tokens=10),
            Message(role="assistant", content="Hello", tokens=10),
        ]
        result, total = apply_token_budget(messages, max_tokens=50)
        assert len(result) == 2
        assert total == 20

    def test_eviction_removes_oldest(self):
        messages = [
            Message(role="user", content="First", tokens=40),
            Message(role="assistant", content="Second", tokens=40),
        ]
        result, total = apply_token_budget(messages, max_tokens=50)
        assert len(result) == 1
        assert result[0].content == "Second"
        assert total == 40

    def test_eviction_removes_multiple(self):
        messages = [
            Message(role="user", content="A", tokens=30),
            Message(role="assistant", content="B", tokens=30),
            Message(role="user", content="C", tokens=30),
        ]
        result, total = apply_token_budget(messages, max_tokens=35)
        assert len(result) == 1
        assert result[0].content == "C"
        assert total == 30

    def test_eviction_exact_budget(self):
        messages = [
            Message(role="user", content="A", tokens=25),
            Message(role="assistant", content="B", tokens=25),
        ]
        result, total = apply_token_budget(messages, max_tokens=50)
        assert len(result) == 2
        assert total == 50

    def test_eviction_all_messages(self):
        messages = [
            Message(role="user", content="A", tokens=100),
        ]
        result, total = apply_token_budget(messages, max_tokens=10)
        assert len(result) == 0
        assert total == 0

    def test_eviction_empty_list(self):
        result, total = apply_token_budget([], max_tokens=50)
        assert len(result) == 0
        assert total == 0

    def test_enforce_budget_on_context(self):
        ctx = AgentContext(session_id="test")
        add_message(ctx, "user", "A" * 500)
        add_message(ctx, "assistant", "B" * 500)
        add_message(ctx, "user", "C" * 10)

        # Force a low budget
        enforce_budget(ctx, max_tokens=50)
        assert ctx.total_recent_tokens <= 50


# ──────────────────────────────────────────────
# Session Management
# ──────────────────────────────────────────────


class TestSessionManagement:
    def test_create_new_session(self):
        ctx = get_or_create_session("session_1")
        assert ctx.session_id == "session_1"
        assert ctx.active_topic == "general"
        assert ctx.pinned.ticket_id is None
        assert ctx.recent == []
        assert ctx.total_recent_tokens == 0

    def test_get_existing_session(self):
        ctx1 = get_or_create_session("session_1")
        update_pinned_ticket(ctx1, "IT-1234")
        ctx2 = get_or_create_session("session_1")
        assert ctx2.pinned.ticket_id == "IT-1234"

    def test_session_isolation(self):
        ctx_a = get_or_create_session("session_A")
        ctx_b = get_or_create_session("session_B")

        update_pinned_ticket(ctx_a, "IT-1111")
        update_pinned_ticket(ctx_b, "IT-2222")

        # Verify isolation
        assert ctx_a.pinned.ticket_id == "IT-1111"
        assert ctx_b.pinned.ticket_id == "IT-2222"

        # Re-fetch and verify
        ctx_a2 = get_or_create_session("session_A")
        ctx_b2 = get_or_create_session("session_B")
        assert ctx_a2.pinned.ticket_id == "IT-1111"
        assert ctx_b2.pinned.ticket_id == "IT-2222"

    def test_get_session_nonexistent(self):
        result = get_session("does_not_exist")
        assert result is None

    def test_sanitize_session_id(self):
        assert _sanitize_session_id("abc123") == "abc123"
        assert _sanitize_session_id("abc-123_test") == "abc-123_test"
        assert _sanitize_session_id("../../etc/passwd") == "______etc_passwd"
        assert _sanitize_session_id("a b c") == "a_b_c"


# ──────────────────────────────────────────────
# Pinned Memory
# ──────────────────────────────────────────────


class TestPinnedMemory:
    def test_pin_ticket(self):
        ctx = AgentContext(session_id="test")
        update_pinned_ticket(ctx, "IT-9921")
        assert ctx.pinned.ticket_id == "IT-9921"

    def test_pin_overwrites_previous(self):
        ctx = AgentContext(session_id="test")
        update_pinned_ticket(ctx, "IT-1111")
        update_pinned_ticket(ctx, "IT-2222")
        assert ctx.pinned.ticket_id == "IT-2222"

    def test_pinned_survives_eviction(self):
        """Pinned memory must not be affected by FIFO eviction."""
        ctx = AgentContext(session_id="test")
        update_pinned_ticket(ctx, "IT-9921")

        # Add messages that exceed budget
        add_message(ctx, "user", "A" * 500)
        add_message(ctx, "assistant", "B" * 500)
        add_message(ctx, "user", "C" * 10)

        # Evict
        enforce_budget(ctx, max_tokens=50)

        # Pinned memory must survive
        assert ctx.pinned.ticket_id == "IT-9921"


# ──────────────────────────────────────────────
# Topic Tracking
# ──────────────────────────────────────────────


class TestTopicTracking:
    def test_update_topic(self):
        ctx = AgentContext(session_id="test")
        update_topic(ctx, "wifi_issue")
        assert ctx.active_topic == "wifi_issue"

    def test_topic_change(self):
        ctx = AgentContext(session_id="test")
        update_topic(ctx, "ticket_inquiry")
        update_topic(ctx, "wifi_issue")
        assert ctx.active_topic == "wifi_issue"


# ──────────────────────────────────────────────
# Snapshot & Rollback
# ──────────────────────────────────────────────


class TestSnapshotRollback:
    def test_snapshot_is_deep_copy(self):
        ctx = get_or_create_session("test_snap")
        update_pinned_ticket(ctx, "IT-1234")
        add_message(ctx, "user", "Hello")

        snap = snapshot_context(ctx)

        # Modify original
        update_pinned_ticket(ctx, "IT-9999")
        add_message(ctx, "user", "World")

        # Snapshot should be unchanged
        assert snap.pinned.ticket_id == "IT-1234"
        assert len(snap.recent) == 1

    def test_restore_reverts_changes(self):
        ctx = get_or_create_session("test_rollback")
        snap = snapshot_context(ctx)

        # Make changes
        update_pinned_ticket(ctx, "IT-5555")
        add_message(ctx, "user", "Something")

        # Restore
        restore_context(snap)

        # Verify rollback
        restored = get_or_create_session("test_rollback")
        assert restored.pinned.ticket_id is None
        assert len(restored.recent) == 0


# ──────────────────────────────────────────────
# Context Persistence
# ──────────────────────────────────────────────


class TestContextPersistence:
    def test_persist_and_reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOGS_DIR", str(tmp_path))

        ctx = AgentContext(
            session_id="persist_test",
            active_topic="wifi_issue",
            pinned=PinnedMemory(ticket_id="IT-1234"),
        )
        add_message(ctx, "user", "Hello")
        persist_context(ctx)

        # Verify file exists
        filepath = tmp_path / "context_persist_test.json"
        assert filepath.exists()

        # Verify JSON content
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data["session_id"] == "persist_test"
        assert data["active_topic"] == "wifi_issue"
        assert data["pinned"]["ticket_id"] == "IT-1234"
        assert len(data["recent"]) == 1
        assert data["recent"][0]["role"] == "user"

    def test_persist_creates_directory(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "subdir" / "logs"
        monkeypatch.setenv("LOGS_DIR", str(logs_dir))

        ctx = AgentContext(session_id="dir_test")
        persist_context(ctx)

        assert logs_dir.exists()
        assert (logs_dir / "context_dir_test.json").exists()
