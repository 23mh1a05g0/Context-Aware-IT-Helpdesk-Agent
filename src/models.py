"""
Pydantic data models for the helpdesk agent.

These models define the core data structures used throughout
the application: messages, context state, and API contracts.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# --------------------------------------------------
# Internal Domain Models
# --------------------------------------------------

class Message(BaseModel):
    """A single conversation message with its token count."""
    role: str = Field(..., pattern=r"^(user|assistant)$", description="Message role")
    content: str = Field(..., min_length=1, description="Message content")
    tokens: int = Field(0, ge=0, description="Token count for this message")


class PinnedMemory(BaseModel):
    """
    Facts that must survive token eviction.
    These are never removed by the FIFO eviction algorithm.
    """
    ticket_id: Optional[str] = Field(None, description="Ticket ID in IT-XXXX format")


class AgentContext(BaseModel):
    """
    Complete context state for a single session.
    This is the core data structure that the context engine manages.
    """
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    active_topic: str = Field("general", description="Current conversation topic")
    pinned: PinnedMemory = Field(default_factory=PinnedMemory)
    recent: list[Message] = Field(default_factory=list)
    total_recent_tokens: int = Field(0, ge=0, description="Sum of tokens in recent messages")


# --------------------------------------------------
# API Request / Response Models
# --------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat request from a client."""
    session_id: str = Field(..., min_length=1, description="Session identifier")
    message: str = Field(..., min_length=1, description="User message text")


class ChatResponse(BaseModel):
    """Chat response returned to the client."""
    response: str = Field(..., description="Assistant response text")
    active_topic: str = Field(..., description="Current conversation topic")


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"


class ErrorResponse(BaseModel):
    """Generic error response."""
    detail: str
