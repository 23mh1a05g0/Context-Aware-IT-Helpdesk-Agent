"""
Prompt builder module.

Constructs the final prompt payload (list of messages) that is sent
to the LLM. Injects:
  1. System persona and instructions
  2. Active topic
  3. Pinned facts
  4. Recent conversation history
"""

from __future__ import annotations

from src.models import AgentContext


_SYSTEM_TEMPLATE = """You are a university IT helpdesk assistant.

Help students and faculty troubleshoot IT issues including WiFi connectivity, \
printing problems, network configuration, password resets, software installation, \
and general IT questions.

Current conversation topic:
{topic}

Known pinned facts:
{pinned_facts}

Instructions:
- Use the pinned facts when relevant to answer questions.
- Do not invent ticket information or claim to know a ticket's live status.
- If the user asks about a ticket but only the ticket ID is known, explain that \
you know the ticket ID from the conversation but do not have access to the \
university's ticketing system to retrieve its live status.
- Be helpful, concise, and professional.
- When troubleshooting, provide step-by-step instructions.
- If the user's operating system is known from pinned facts, tailor instructions accordingly."""


def _format_pinned_facts(ctx: AgentContext) -> str:
    """Format pinned memory into a readable string for the system prompt."""
    facts: list[str] = []

    if ctx.pinned.ticket_id:
        facts.append(f"Ticket ID: {ctx.pinned.ticket_id}")

    if not facts:
        return "None"

    return "\n".join(facts)


def build_prompt(ctx: AgentContext) -> list[dict[str, str]]:
    """
    Build the complete prompt payload for the LLM.

    Returns a list of message dicts in the OpenAI chat format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]

    Args:
        ctx: The current session context.

    Returns:
        List of message dictionaries.
    """
    pinned_facts = _format_pinned_facts(ctx)

    system_content = _SYSTEM_TEMPLATE.format(
        topic=ctx.active_topic,
        pinned_facts=pinned_facts,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]

    for msg in ctx.recent:
        messages.append({"role": msg.role, "content": msg.content})

    return messages
