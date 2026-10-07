"""
Tests for LLM failure rollback behavior.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.context_engine import (
    clear_all_sessions,
    get_or_create_session,
    get_session,
    update_pinned_ticket,
)
from src.llm_client import LLMError, MockLLMClient, reset_llm_client


@pytest.fixture(autouse=True)
def clean_state():
    clear_all_sessions(clear_disk=True)
    reset_llm_client()
    yield
    clear_all_sessions(clear_disk=True)
    reset_llm_client()


class TestErrorRollback:
    """Tests for error rollback on LLM failure."""

    @pytest.mark.anyio
    async def test_rollback_on_llm_failure(self):
        """When LLM fails, context must be restored to pre-request state."""
        from httpx import ASGITransport, AsyncClient
        from src.api import app

        # Force mock LLM
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # First: successful request to establish session
                resp = await client.post("/v1/chat", json={
                    "session_id": "rollback_test",
                    "message": "My ticket is IT-5555"
                })
                assert resp.status_code == 200

                # Verify ticket was pinned
                ctx_resp = await client.get("/v1/context/rollback_test")
                assert ctx_resp.json()["pinned"]["ticket_id"] == "IT-5555"
                messages_before = len(ctx_resp.json()["recent"])

                # Now simulate LLM failure
                failing_client = MockLLMClient()
                failing_client.generate = AsyncMock(
                    side_effect=LLMError("Simulated failure")
                )

                with patch("src.api.get_llm_client", return_value=failing_client):
                    resp = await client.post("/v1/chat", json={
                        "session_id": "rollback_test",
                        "message": "This should fail"
                    })
                    assert resp.status_code == 500

                # Verify context was rolled back — the failed message should NOT
                # appear in recent history
                ctx_resp = await client.get("/v1/context/rollback_test")
                ctx = ctx_resp.json()
                assert ctx["pinned"]["ticket_id"] == "IT-5555"
                assert len(ctx["recent"]) == messages_before

                # The failed message "This should fail" should NOT be in recent
                contents = [m["content"] for m in ctx["recent"]]
                assert "This should fail" not in contents

    @pytest.mark.anyio
    async def test_rollback_preserves_empty_state(self):
        """LLM failure on first message should leave session empty."""
        from httpx import ASGITransport, AsyncClient
        from src.api import app

        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                failing_client = MockLLMClient()
                failing_client.generate = AsyncMock(
                    side_effect=LLMError("Simulated failure")
                )

                with patch("src.api.get_llm_client", return_value=failing_client):
                    resp = await client.post("/v1/chat", json={
                        "session_id": "empty_rollback",
                        "message": "Hello"
                    })
                    assert resp.status_code == 500

                # Session should exist but be empty
                ctx_resp = await client.get("/v1/context/empty_rollback")
                ctx = ctx_resp.json()
                assert ctx["recent"] == []
                assert ctx["total_recent_tokens"] == 0

    @pytest.mark.anyio
    async def test_error_response_does_not_leak_details(self):
        """Error responses must not expose internal details."""
        from httpx import ASGITransport, AsyncClient
        from src.api import app

        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                failing_client = MockLLMClient()
                failing_client.generate = AsyncMock(
                    side_effect=LLMError("SECRET_API_KEY_12345")
                )

                with patch("src.api.get_llm_client", return_value=failing_client):
                    resp = await client.post("/v1/chat", json={
                        "session_id": "leak_test",
                        "message": "Hello"
                    })
                    assert resp.status_code == 500
                    body = resp.json()
                    # Must not contain the secret
                    assert "SECRET_API_KEY_12345" not in body.get("detail", "")
