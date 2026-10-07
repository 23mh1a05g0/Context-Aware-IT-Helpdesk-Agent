"""
Five-Turn Context Evaluation Script

Runs a complete 5-turn conversation through the API to verify:
  A. Ticket ID extraction
  B. Pinned memory storage
  C. Topic switching
  D. Token budget eviction
  E. Pinned memory survival after eviction
  F. Ticket ID in final LLM response
  G. Context persistence

Usage:
    python eval_5_turns.py

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure UTF-8 output if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Force mock mode and low token budget for eviction testing
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("TOKEN_BUDGET_LIMIT", "100")
os.environ.setdefault("LOGS_DIR", "logs")

import httpx
from httpx import ASGITransport, AsyncClient

from src.api import app
from src.context_engine import clear_all_sessions
from src.llm_client import reset_llm_client


SESSION_ID = "eval_session_5turn"

TURNS = [
    {
        "message": "Hi, I need help.",
        "checks": {
            "active_topic": "general",
        },
    },
    {
        "message": "My ticket is IT-8812.",
        "checks": {
            "pinned_ticket": "IT-8812",
            "active_topic": "ticket_inquiry",
        },
    },
    {
        "message": (
            "Actually, before we do that, my WiFi in the library just dropped. "
            "I am using a Mac. It says connected without internet. "
            "Please give me detailed troubleshooting instructions."
        ),
        "checks": {
            "active_topic": "wifi_issue",
            "pinned_ticket": "IT-8812",  # Must survive topic switch
        },
    },
    {
        "message": "Can you give me more detailed Mac troubleshooting steps?",
        "checks": {
            "active_topic": "wifi_issue",
            "pinned_ticket": "IT-8812",
        },
    },
    {
        "message": (
            "Okay, I fixed the WiFi. Let's go back to my original ticket. "
            "Can you confirm the ticket number I mentioned earlier?"
        ),
        "checks": {
            "pinned_ticket": "IT-8812",
            "response_contains": "IT-8812",
        },
    },
]


class EvalResult:
    def __init__(self):
        self.results: list[tuple[str, bool]] = []

    def check(self, label: str, passed: bool) -> None:
        self.results.append((label, passed))
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {label}")

    @property
    def all_passed(self) -> bool:
        return all(p for _, p in self.results)


async def run_evaluation() -> bool:
    """Run the 5-turn evaluation. Returns True if all checks pass."""
    clear_all_sessions()
    reset_llm_client()

    # Clean up previous evaluation log file if present
    log_file = os.path.join(
        os.environ.get("LOGS_DIR", "logs"),
        f"context_{SESSION_ID}.json"
    )
    if os.path.exists(log_file):
        os.remove(log_file)

    result = EvalResult()

    print()
    print("=" * 50)
    print("  5-TURN CONTEXT EVALUATION")
    print("=" * 50)
    print()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        # -- Health check --
        health = await client.get("/health")
        result.check("Health endpoint returns 200", health.status_code == 200)

        # -- Run turns --
        for i, turn in enumerate(TURNS, 1):
            print(f"\n  -- Turn {i} --")
            print(f"  User: {turn['message'][:80]}{'...' if len(turn['message']) > 80 else ''}")

            resp = await client.post("/v1/chat", json={
                "session_id": SESSION_ID,
                "message": turn["message"],
            })

            result.check(f"Turn {i}: HTTP 200", resp.status_code == 200)

            if resp.status_code != 200:
                print(f"    ERROR: {resp.text}")
                continue

            chat_data = resp.json()
            print(f"  Assistant: {chat_data['response'][:80]}...")
            print(f"  Topic: {chat_data['active_topic']}")

            # Get context
            ctx_resp = await client.get(f"/v1/context/{SESSION_ID}")
            ctx = ctx_resp.json()

            checks = turn["checks"]

            if "active_topic" in checks:
                result.check(
                    f"Turn {i}: active_topic == {checks['active_topic']}",
                    ctx["active_topic"] == checks["active_topic"]
                )

            if "pinned_ticket" in checks:
                result.check(
                    f"Turn {i}: pinned ticket == {checks['pinned_ticket']}",
                    ctx["pinned"]["ticket_id"] == checks["pinned_ticket"]
                )

            if "response_contains" in checks:
                result.check(
                    f"Turn {i}: response contains {checks['response_contains']}",
                    checks["response_contains"] in chat_data["response"]
                )

        # -- Post-evaluation checks --
        print(f"\n  -- Post-Evaluation Checks --")

        ctx_resp = await client.get(f"/v1/context/{SESSION_ID}")
        ctx = ctx_resp.json()

        # Token eviction check
        result.check(
            "Token eviction occurred (recent_tokens <= budget)",
            ctx["total_recent_tokens"] <= int(os.environ.get("TOKEN_BUDGET_LIMIT", "100"))
        )

        # Pinned memory survived eviction
        result.check(
            "Pinned ticket survived eviction (IT-8812)",
            ctx["pinned"]["ticket_id"] == "IT-8812"
        )

        # Check that context was persisted
        log_file = os.path.join(
            os.environ.get("LOGS_DIR", "logs"),
            f"context_{SESSION_ID}.json"
        )
        persisted = os.path.exists(log_file)
        result.check("Context persisted to disk", persisted)

        if persisted:
            with open(log_file, "r", encoding="utf-8") as f:
                persisted_data = json.load(f)
            result.check(
                "Persisted context has correct ticket",
                persisted_data["pinned"]["ticket_id"] == "IT-8812"
            )

        # Check that old messages were evicted (budget is 100 tokens)
        all_contents = [m["content"] for m in ctx["recent"]]
        original_ticket_msg = "My ticket is IT-8812."
        ticket_msg_evicted = original_ticket_msg not in all_contents
        result.check(
            "Original ticket message evicted from recent (proves eviction)",
            ticket_msg_evicted
        )

    # -- Summary --
    print()
    print("=" * 50)
    passed = sum(1 for _, p in result.results if p)
    total = len(result.results)

    if result.all_passed:
        print(f"  RESULT: PASS ({passed}/{total} checks)")
    else:
        print(f"  RESULT: FAIL ({passed}/{total} checks passed)")
        for label, p in result.results:
            if not p:
                print(f"    FAILED: {label}")

    print("=" * 50)
    print()

    # Print final context state
    print("  Final Context State:")
    print(f"  {json.dumps(ctx, indent=2)}")
    print()

    return result.all_passed


def main():
    success = asyncio.run(run_evaluation())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
