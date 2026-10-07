"""
Tests for the FastAPI API endpoints.
"""

import json
import pytest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from src.api import app
from src.context_engine import clear_all_sessions
from src.llm_client import reset_llm_client


@pytest.fixture(autouse=True)
def clean_state():
    clear_all_sessions(clear_disk=True)
    reset_llm_client()
    yield
    clear_all_sessions(clear_disk=True)
    reset_llm_client()


# ──────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.anyio
    async def test_health_returns_ok(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


# ──────────────────────────────────────────────
# Chat Endpoint
# ──────────────────────────────────────────────


class TestChatEndpoint:
    @pytest.mark.anyio
    async def test_chat_basic(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat", json={
                    "session_id": "test_chat",
                    "message": "Hello"
                })
                assert resp.status_code == 200
                data = resp.json()
                assert "response" in data
                assert "active_topic" in data

    @pytest.mark.anyio
    async def test_chat_extracts_ticket(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat", json={
                    "session_id": "ticket_test",
                    "message": "My ticket is IT-9921"
                })
                assert resp.status_code == 200
                data = resp.json()
                assert data["active_topic"] == "ticket_inquiry"

                # Verify context
                ctx_resp = await client.get("/v1/context/ticket_test")
                assert ctx_resp.status_code == 200
                ctx = ctx_resp.json()
                assert ctx["pinned"]["ticket_id"] == "IT-9921"

    @pytest.mark.anyio
    async def test_chat_topic_detection(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat", json={
                    "session_id": "topic_test",
                    "message": "My WiFi is not working"
                })
                assert resp.status_code == 200
                assert resp.json()["active_topic"] == "wifi_issue"

    @pytest.mark.anyio
    async def test_chat_missing_session_id(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/chat", json={
                "message": "Hello"
            })
            assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_chat_missing_message(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/chat", json={
                "session_id": "test"
            })
            assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_chat_empty_session_id(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/chat", json={
                "session_id": "",
                "message": "Hello"
            })
            assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_chat_empty_message(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/chat", json={
                "session_id": "test",
                "message": ""
            })
            assert resp.status_code == 422


# ──────────────────────────────────────────────
# Context Endpoint
# ──────────────────────────────────────────────


class TestContextEndpoint:
    @pytest.mark.anyio
    async def test_context_returns_state(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Create session via chat
                await client.post("/v1/chat", json={
                    "session_id": "ctx_test",
                    "message": "My ticket is IT-1234"
                })

                resp = await client.get("/v1/context/ctx_test")
                assert resp.status_code == 200
                data = resp.json()
                assert data["session_id"] == "ctx_test"
                assert data["pinned"]["ticket_id"] == "IT-1234"
                assert "recent" in data
                assert "total_recent_tokens" in data

    @pytest.mark.anyio
    async def test_context_404_for_missing_session(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/context/nonexistent_session")
            assert resp.status_code == 404


# ──────────────────────────────────────────────
# Session Isolation via API
# ──────────────────────────────────────────────


class TestSessionIsolationAPI:
    @pytest.mark.anyio
    async def test_sessions_are_isolated(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Session A
                await client.post("/v1/chat", json={
                    "session_id": "session_A",
                    "message": "My ticket is IT-1111"
                })

                # Session B
                await client.post("/v1/chat", json={
                    "session_id": "session_B",
                    "message": "My ticket is IT-2222"
                })

                # Verify isolation
                ctx_a = await client.get("/v1/context/session_A")
                ctx_b = await client.get("/v1/context/session_B")

                assert ctx_a.json()["pinned"]["ticket_id"] == "IT-1111"
                assert ctx_b.json()["pinned"]["ticket_id"] == "IT-2222"


# ──────────────────────────────────────────────
# Multi-Turn Conversation via API
# ──────────────────────────────────────────────


class TestMultiTurnConversation:
    @pytest.mark.anyio
    async def test_five_turn_conversation(self):
        """Full five-turn conversation test through the API."""
        with patch.dict("os.environ", {
            "LLM_PROVIDER": "mock",
            "TOKEN_BUDGET_LIMIT": "100"
        }):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                sid = "five_turn_test"

                # Turn 1: Greeting
                r1 = await client.post("/v1/chat", json={
                    "session_id": sid,
                    "message": "Hi, I need help."
                })
                assert r1.status_code == 200
                assert r1.json()["active_topic"] == "general"

                # Turn 2: Ticket
                r2 = await client.post("/v1/chat", json={
                    "session_id": sid,
                    "message": "My ticket is IT-8812."
                })
                assert r2.status_code == 200
                ctx2 = await client.get(f"/v1/context/{sid}")
                assert ctx2.json()["pinned"]["ticket_id"] == "IT-8812"
                assert ctx2.json()["active_topic"] == "ticket_inquiry"

                # Turn 3: WiFi
                r3 = await client.post("/v1/chat", json={
                    "session_id": sid,
                    "message": "Actually, before we do that, my WiFi in the library just dropped. I am using a Mac. It says connected without internet. Please give me detailed troubleshooting instructions."
                })
                assert r3.status_code == 200
                ctx3 = await client.get(f"/v1/context/{sid}")
                assert ctx3.json()["active_topic"] == "wifi_issue"
                assert ctx3.json()["pinned"]["ticket_id"] == "IT-8812"

                # Turn 4: More WiFi help
                r4 = await client.post("/v1/chat", json={
                    "session_id": sid,
                    "message": "Can you give me more detailed Mac troubleshooting steps?"
                })
                assert r4.status_code == 200

                # Turn 5: Return to ticket
                r5 = await client.post("/v1/chat", json={
                    "session_id": sid,
                    "message": "Okay, I fixed the WiFi. Let's go back to my original ticket. Can you confirm the ticket number I mentioned earlier?"
                })
                assert r5.status_code == 200
                assert "IT-8812" in r5.json()["response"]

                # Final verification
                ctx_final = await client.get(f"/v1/context/{sid}")
                ctx = ctx_final.json()
                assert ctx["pinned"]["ticket_id"] == "IT-8812"

    @pytest.mark.anyio
    async def test_very_long_message(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "mock"}):
            reset_llm_client()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                long_message = "Help me with WiFi. " * 500
                resp = await client.post("/v1/chat", json={
                    "session_id": "long_msg_test",
                    "message": long_message
                })
                assert resp.status_code == 200
