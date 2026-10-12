"""Validate and authorize every task in an execution plan."""

from __future__ import annotations

from packages.agents.planning.plan_validator import validate_execution_plan
from packages.agents.state import AgentState
from packages.authz.audit import write_audit_log
from packages.observability.repositories.execution_plan_repo import (
    finalize_execution_plan,
    persist_validated_plan,
)
from packages.rag.prompts import NO_PERMISSION_ANSWER


async def plan_validator_node(state: AgentState) -> AgentState:
    plan = state["execution_plan"]
    plan = await validate_execution_plan(
        state["db_session"],
        state["user_context"],
        plan,
        session_id=state.get("session_id") or state["trace_id"],
        use_memory=state.get("use_memory", True),
    )
    user_context = state["user_context"]
    await write_audit_log(
        state["db_session"],
        tenant_id=user_context.tenant_id,
        user_id=user_context.user_id,
        trace_id=state["trace_id"],
        action="execution_plan_validated",
        resource_type="agent_plan",
        resource_id=plan.plan_id,
        permission_decision="denied" if plan.status == "blocked" else "allowed",
        reason=f"status={plan.status}; tasks={len(plan.tasks)}",
        ip_address=state.get("ip_address"),
        metadata={
            "tasks": [task.model_dump(mode="json") for task in plan.tasks],
            "validation_errors": plan.validation_errors,
        },
    )
    await persist_validated_plan(
        state["db_session"],
        plan,
        agent_run_id=state["agent_run_id"],
        tenant_id=user_context.tenant_id,
    )
    if plan.status in {"blocked", "clarification_required"}:
        await finalize_execution_plan(state["db_session"], plan)
    updates: AgentState = {
        "execution_plan": plan,
        "plan_status": plan.status,
        "clarification_question": plan.clarification_question,
        "nodes_executed": ["plan_validator"],
    }
    if plan.status == "blocked":
        updates.update(
            {
                "status": "blocked",
                "draft_answer": NO_PERMISSION_ANSWER,
                "final_answer": NO_PERMISSION_ANSWER,
            }
        )
    return updates
