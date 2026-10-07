"""
Tests specifically for token eviction behavior.
"""

import pytest
from src.context_engine import (
    add_message,
    apply_token_budget,
    count_tokens,
    enforce_budget,
    update_pinned_ticket,
)
from src.models import AgentContext, Message


class TestFIFOEviction:
    """Focused tests on FIFO token eviction algorithm."""

    def test_basic_eviction(self):
        """Total > budget → oldest removed."""
        messages = [
            Message(role="user", content="Message A", tokens=40),
            Message(role="assistant", content="Message B", tokens=40),
        ]
        result, total = apply_token_budget(messages, max_tokens=50)
        assert len(result) == 1
        assert result[0].content == "Message B"
        assert total == 40

    def test_no_eviction_under_budget(self):
        messages = [
            Message(role="user", content="Hi", tokens=5),
            Message(role="assistant", content="Hello", tokens=5),
        ]
        result, total = apply_token_budget(messages, max_tokens=100)
        assert len(result) == 2
        assert total == 10

    def test_cascading_eviction(self):
        """Multiple messages must be removed to fit budget."""
        messages = [
            Message(role="user", content="A", tokens=30),
            Message(role="assistant", content="B", tokens=30),
            Message(role="user", content="C", tokens=30),
            Message(role="assistant", content="D", tokens=30),
        ]
        result, total = apply_token_budget(messages, max_tokens=65)
        assert len(result) == 2
        assert result[0].content == "C"
        assert result[1].content == "D"
        assert total == 60

    def test_eviction_preserves_order(self):
        messages = [
            Message(role="user", content="1", tokens=20),
            Message(role="assistant", content="2", tokens=20),
            Message(role="user", content="3", tokens=20),
            Message(role="assistant", content="4", tokens=20),
            Message(role="user", content="5", tokens=20),
        ]
        result, total = apply_token_budget(messages, max_tokens=45)
        assert [m.content for m in result] == ["4", "5"]

    def test_eviction_all_removed(self):
        """If every message exceeds budget individually, all are removed."""
        messages = [
            Message(role="user", content="Big", tokens=100),
            Message(role="assistant", content="Bigger", tokens=100),
        ]
        result, total = apply_token_budget(messages, max_tokens=10)
        assert len(result) == 0
        assert total == 0

    def test_single_message_over_budget(self):
        messages = [Message(role="user", content="Huge", tokens=200)]
        result, total = apply_token_budget(messages, max_tokens=50)
        assert len(result) == 0
        assert total == 0

    def test_original_list_not_mutated(self):
        """apply_token_budget must not mutate the input list."""
        messages = [
            Message(role="user", content="A", tokens=60),
            Message(role="assistant", content="B", tokens=60),
        ]
        original_len = len(messages)
        apply_token_budget(messages, max_tokens=70)
        assert len(messages) == original_len


class TestEvictionWithPinnedMemory:
    """Verify pinned memory is completely independent of eviction."""

    def test_pinned_ticket_survives_full_eviction(self):
        ctx = AgentContext(session_id="test")
        update_pinned_ticket(ctx, "IT-9921")

        # Fill with messages
        for i in range(10):
            add_message(ctx, "user", f"Message {i} " * 50)
            add_message(ctx, "assistant", f"Reply {i} " * 50)

        # Aggressive eviction
        enforce_budget(ctx, max_tokens=20)

        # Pinned memory MUST survive
        assert ctx.pinned.ticket_id == "IT-9921"
        assert ctx.total_recent_tokens <= 20

    def test_pinned_survives_while_recent_cleared(self):
        ctx = AgentContext(session_id="test")
        update_pinned_ticket(ctx, "IT-8812")

        # Add enough to exceed a tiny budget
        add_message(ctx, "user", "A very long message " * 100)

        enforce_budget(ctx, max_tokens=5)

        assert ctx.pinned.ticket_id == "IT-8812"
        # recent might be empty but pinned is safe
        assert len(ctx.recent) == 0 or ctx.total_recent_tokens <= 5


class TestTokenCountingAccuracy:
    """Verify tiktoken integration correctness."""

    def test_known_token_count(self):
        # "Hello" in cl100k_base is 1 token
        assert count_tokens("Hello") >= 1

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_long_text_has_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("Hello world, this is a longer sentence with many words")
        assert long > short

    def test_special_characters(self):
        """Special characters should still tokenize."""
        count = count_tokens("IT-9921 @#$% !!! ???")
        assert count > 0
