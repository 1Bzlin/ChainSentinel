"""Create an execution plan for the current request."""

from __future__ import annotations

from packages.agents.planning.plan_builder import build_execution_plan
from packages.agents.state import AgentState
from packages.authz.audit import write_audit_log
from packages.observability.repositories.execution_plan_repo import (
    create_execution_plan,
)


async def planner_node(state: AgentState) -> AgentState:
    original_query = state.get("original_query") or state["query"]
    resolved_query = state.get("resolved_query") or original_query
    plan = await build_execution_plan(
        original_query=original_query,
        resolved_query=resolved_query,
        session_context=state.get("session_context") or {},
        resolved_entities=state.get("resolved_entities") or {},
        context_references=state.get("context_references") or [],
        context_unresolved=bool(state.get("context_unresolved")),
        top_k=state.get("top_k", 5),
        doc_type=state.get("doc_type"),
        session_id=state.get("session_id") or state["trace_id"],
        use_memory=state.get("use_memory", True),
    )
    user_context = state["user_context"]
    await write_audit_log(
        state["db_session"],
        tenant_id=user_context.tenant_id,
        user_id=user_context.user_id,
        trace_id=state["trace_id"],
        action="execution_plan_created",
        resource_type="agent_plan",
        resource_id=plan.plan_id,
        permission_decision="allowed",
        reason=f"source={plan.source}; tasks={len(plan.tasks)}; confidence={plan.confidence:.2f}",
        ip_address=state.get("ip_address"),
        metadata=plan.model_dump(mode="json"),
    )
    await create_execution_plan(
        state["db_session"],
        plan,
        agent_run_id=state["agent_run_id"],
        tenant_id=user_context.tenant_id,
    )
    return {
        "query": plan.resolved_query,
        "resolved_query": plan.resolved_query,
        "resolved_entities": plan.entities,
        "execution_plan": plan,
        "plan_status": plan.status,
        "planning_source": plan.source,
        "planning_confidence": plan.confidence,
        "planning_token_usage": plan.token_usage,
        "clarification_question": plan.clarification_question,
        "nodes_executed": ["planner"],
    }
