"""Permission node."""

from __future__ import annotations

import uuid

from packages.agents.planning.schemas import ExecutionPlan
from packages.agents.state import AgentState
from packages.authz.audit import write_audit_log
from packages.authz.rbac import PermissionDecision
from packages.rag.prompts import NO_PERMISSION_ANSWER
from packages.observability.repositories.execution_plan_repo import (
    create_execution_plan,
    finalize_execution_plan,
)


async def permission_node(state: AgentState) -> AgentState:
    user_context = state["user_context"]
    session = state["db_session"]
    trace_id = state["trace_id"]

    if not user_context:
        decision = PermissionDecision(False, "missing user context")
    elif user_context.is_external_supplier and not user_context.supplier_id:
        decision = PermissionDecision(False, "external supplier missing supplier_id")
    else:
        decision = PermissionDecision(True, "query permitted for authenticated user")

    await write_audit_log(
        session,
        tenant_id=user_context.tenant_id,
        user_id=user_context.user_id,
        trace_id=trace_id,
        action=(
            "chat_permission_allowed" if decision.allowed else "chat_permission_denied"
        ),
        resource_type="chat",
        resource_id=trace_id,
        permission_decision="allowed" if decision.allowed else "denied",
        reason=decision.reason,
        ip_address=state.get("ip_address"),
        metadata={"query": state["query"]},
    )

    updates: AgentState = {
        "permission_decision": decision,
        "nodes_executed": ["permission"],
    }
    if not decision.allowed:
        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            original_query=state["query"],
            resolved_query=state["query"],
            primary_intent="permission_denied",
            source="system",
            confidence=1.0,
            status="blocked",
            tasks=[],
            validation_errors=[decision.reason],
        )
        updates["draft_answer"] = NO_PERMISSION_ANSWER
        updates["final_answer"] = NO_PERMISSION_ANSWER
        updates["status"] = "blocked"
        updates["review_passed"] = True
        updates["execution_plan"] = plan
        updates["plan_status"] = "blocked"
        updates["planning_source"] = "system"
        updates["planning_confidence"] = 1.0
        updates["plan_summary"] = []
        await create_execution_plan(
            session,
            plan,
            agent_run_id=state["agent_run_id"],
            tenant_id=user_context.tenant_id,
        )
        await finalize_execution_plan(session, plan)
    return updates
