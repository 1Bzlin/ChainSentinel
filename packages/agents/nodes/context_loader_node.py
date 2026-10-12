"""Load compact conversational context before routing."""

from __future__ import annotations

from packages.agents.planning.context_resolver import (
    build_session_context,
    resolve_query_context,
)
from packages.agents.state import AgentState
from packages.memory.short_term import get_recent_entities, get_recent_messages


async def context_loader_node(state: AgentState) -> AgentState:
    if state.get("permission_decision") and not state["permission_decision"].allowed:
        return {"nodes_executed": ["context_loader_skipped"]}

    user_context = state["user_context"]
    session_id = state.get("session_id") or state["trace_id"]
    tenant_id = str(user_context.tenant_id)
    user_id = str(user_context.user_id)

    try:
        messages = await get_recent_messages(tenant_id, user_id, session_id, limit=6)
        recent_entities = await get_recent_entities(
            tenant_id, user_id, session_id, limit=10
        )
    except Exception:
        messages = []
        recent_entities = {}

    resolved_query, entities, references, unresolved = resolve_query_context(
        state["query"], recent_entities
    )
    return {
        "original_query": state["query"],
        "resolved_query": resolved_query,
        "recent_entities": recent_entities,
        "resolved_entities": entities,
        "session_context": build_session_context(messages, recent_entities),
        "context_references": references,
        "context_unresolved": unresolved,
        "nodes_executed": ["context_loader"],
    }
