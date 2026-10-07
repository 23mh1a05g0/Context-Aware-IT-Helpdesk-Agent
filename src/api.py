"""
FastAPI application and API endpoints.

Implements:
  POST /v1/chat        -- Main chat endpoint
  GET  /v1/context/{session_id} -- Inspect session context
  GET  /health         -- Health check

The /v1/chat pipeline follows a strict order:
  1. Validate -> 2. Load session -> 3. Snapshot -> 4. Analyze ->
  5. Extract ticket -> 6. Pin -> 7. Detect topic -> 8. Add message ->
  9. Count tokens -> 10. Evict -> 11. Build prompt -> 12. Call LLM ->
  13. Handle success/failure with rollback
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.config import get_token_budget_limit
from src.context_engine import (
    add_message,
    enforce_budget,
    get_or_create_session,
    get_session,
    persist_context,
    restore_context,
    snapshot_context,
    update_pinned_ticket,
    update_topic,
)
from src.extractor import detect_topic, extract_ticket_id
from src.llm_client import LLMError, get_llm_client, reset_llm_client
from src.models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from src.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Application Lifespan
# --------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("IT Helpdesk Agent starting up")
    yield
    reset_llm_client()
    logger.info("IT Helpdesk Agent shutting down")


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Context-Aware IT Helpdesk Agent",
    description="University IT helpdesk with pinned memory, token budgets, and active topic tracking.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# GET /health
# --------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


# --------------------------------------------------
# POST /v1/chat
# --------------------------------------------------


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Pipeline:
      1. Validate request (handled by Pydantic)
      2. Load or create session
      3. Snapshot context for rollback
      4-6. Extract ticket ID -> update pinned memory
      7. Detect active topic
      8. Add user message to recent
      9-10. Apply token budget eviction
      11. Build LLM prompt
      12. Call LLM
      13. On success: append response, persist, return
      14. On failure: rollback, return 500
    """
    # Step 2: Load or create session
    ctx = get_or_create_session(request.session_id)

    # Step 3: Snapshot for rollback
    snapshot = snapshot_context(ctx)

    try:
        # Step 5: Extract ticket ID
        ticket_id = extract_ticket_id(request.message)
        if ticket_id:
            # Step 6: Update pinned memory
            update_pinned_ticket(ctx, ticket_id)

        # Step 7: Detect and update active topic
        topic = detect_topic(request.message)
        update_topic(ctx, topic)

        # Step 8: Add user message to recent
        add_message(ctx, "user", request.message)

        # Steps 9-10: Apply token budget eviction
        enforce_budget(ctx)

        # Step 11: Build prompt
        prompt_messages = build_prompt(ctx)

        # Step 12: Call LLM
        llm = get_llm_client()
        response_text = await llm.generate(prompt_messages)

        # Step 13a: Add assistant response
        add_message(ctx, "assistant", response_text)

        # Apply budget again after assistant message
        enforce_budget(ctx)

        # Persist context
        persist_context(ctx)

        return ChatResponse(response=response_text, active_topic=ctx.active_topic)

    except LLMError as exc:
        # Step 14: Rollback on LLM failure
        logger.error("LLM error, rolling back session %s: %s", request.session_id, exc)
        restore_context(snapshot)
        raise HTTPException(status_code=500, detail="An internal error occurred while generating a response.")

    except Exception as exc:
        # Unexpected error -- still rollback
        logger.error("Unexpected error in chat for session %s: %s", request.session_id, exc)
        restore_context(snapshot)
        raise HTTPException(status_code=500, detail="An internal error occurred.")


# --------------------------------------------------
# GET /v1/context/{session_id}
# --------------------------------------------------


@app.get("/v1/context/{session_id}")
async def get_context(session_id: str) -> dict:
    """
    Retrieve the full context state for a session.

    Returns 404 if the session does not exist.
    """
    ctx = get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return ctx.model_dump()


# --------------------------------------------------
# Global Exception Handler
# --------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all handler to prevent leaking internal details."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
    )
