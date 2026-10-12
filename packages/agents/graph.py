"""LangGraph workflow assembly."""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from packages.agents.nodes.evidence_aggregator_node import evidence_aggregator_node
from packages.agents.nodes.clarify_node import clarify_node
from packages.agents.nodes.context_loader_node import context_loader_node
from packages.agents.nodes.memory_writer_node import memory_writer_node
from packages.agents.nodes.plan_executor_node import plan_executor_node
from packages.agents.nodes.plan_validator_node import plan_validator_node
from packages.agents.nodes.planner_node import planner_node
from packages.agents.nodes.permission_node import permission_node
from packages.agents.nodes.report_node import report_node
from packages.agents.nodes.review_node import review_node
from packages.agents.state import AgentState
from packages.authz.rbac import UserContext
from packages.observability.callbacks import (
    get_langgraph_callbacks,
    setup_langsmith_tracing,
    wrap_agent_node,
)
from packages.observability.repositories.agent_run_repo import (
    create_agent_run,
    finalize_agent_run,
)
from packages.observability.trace_context import (
    get_request_id,
    resolve_trace_id,
    set_trace_id,
)


def _route_after_permission(state: AgentState) -> str:
    decision = state.get("permission_decision")
    if decision and not decision.allowed:
        return "review"
    return "context_loader"


def _route_after_plan_validation(state: AgentState) -> str:
    status = state.get("plan_status")
    if status == "blocked":
        return "review"
    if status == "clarification_required":
        return "clarify"
    return "plan_executor"


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("permission", wrap_agent_node("permission", permission_node))
    graph.add_node(
        "context_loader", wrap_agent_node("context_loader", context_loader_node)
    )
    graph.add_node("planner", wrap_agent_node("planner", planner_node))
    graph.add_node(
        "plan_validator", wrap_agent_node("plan_validator", plan_validator_node)
    )
    graph.add_node(
        "plan_executor", wrap_agent_node("plan_executor", plan_executor_node)
    )
    graph.add_node("clarify", wrap_agent_node("clarify", clarify_node))
    graph.add_node(
        "aggregator", wrap_agent_node("aggregator", evidence_aggregator_node)
    )
    graph.add_node("report", wrap_agent_node("report", report_node))
    graph.add_node("review", wrap_agent_node("review", review_node))
    graph.add_node(
        "memory_writer", wrap_agent_node("memory_writer", memory_writer_node)
    )

    graph.set_entry_point("permission")
    graph.add_conditional_edges(
        "permission",
        _route_after_permission,
        {"context_loader": "context_loader", "review": "review"},
    )
    graph.add_edge("context_loader", "planner")
    graph.add_edge("planner", "plan_validator")
    graph.add_conditional_edges(
        "plan_validator",
        _route_after_plan_validation,
        {
            "plan_executor": "plan_executor",
            "clarify": "clarify",
            "review": "review",
        },
    )
    graph.add_edge("clarify", "review")
    graph.add_edge("plan_executor", "aggregator")
    graph.add_edge("aggregator", "report")
    graph.add_edge("report", "review")
    graph.add_edge("review", "memory_writer")
    graph.add_edge("memory_writer", END)
    return graph.compile()


async def run_chat_workflow(
    session: AsyncSession,
    user_context: UserContext,
    query: str,
    *,
    session_id: str | None = None,
    top_k: int = 5,
    chunk_size_config: int = 512,
    doc_type: str | None = None,
    supplier_id: str | None = None,
    ip_address: str | None = None,
    use_memory: bool = True,
    trace_id: str | None = None,
) -> dict[str, Any]:
    setup_langsmith_tracing()
    trace_id = resolve_trace_id(header_value=trace_id)
    set_trace_id(trace_id)
    session_id = session_id or trace_id
    started = time.perf_counter()

    agent_run = await create_agent_run(
        session,
        tenant_id=user_context.tenant_id,
        user_id=user_context.user_id,
        trace_id=trace_id,
        session_id=session_id,
        query=query,
        workflow_version="planner-v1",
        request_id=get_request_id(),
    )

    initial_state: AgentState = {
        "query": query,
        "user_context": user_context,
        "trace_id": trace_id,
        "session_id": session_id,
        "agent_run_id": str(agent_run.id),
        "top_k": top_k,
        "chunk_size_config": chunk_size_config,
        "doc_type": doc_type,
        "supplier_id": supplier_id,
        "ip_address": ip_address,
        "use_memory": use_memory,
        "db_session": session,
        "errors": [],
        "nodes_executed": [],
        "memory_used": [],
        "memory_written": [],
        "task_results": [],
        "plan_summary": [],
    }

    graph = build_agent_graph()
    callbacks = get_langgraph_callbacks()
    config = {"callbacks": callbacks} if callbacks else {}
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
        latency_ms = int((time.perf_counter() - started) * 1000)
        citations = [item.to_dict() for item in final_state.get("citations") or []]
        status = final_state.get("status") or "completed"
        plan_status = final_state.get("plan_status")
        if status == "clarification_required":
            run_status = "clarification_required"
        elif status == "blocked":
            run_status = "blocked"
        elif plan_status == "failed":
            run_status = "failed"
        elif plan_status == "partial" or status == "partial":
            run_status = "partial"
        else:
            run_status = "completed"

        memory_used = [
            item
            for item in (final_state.get("memory_used") or [])
            if item.get("citation")
        ]

        report_usage = final_state.get("token_usage") or {}
        planning_usage = final_state.get("planning_token_usage") or {}
        token_usage = dict(report_usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in planning_usage:
                token_usage[key] = int(token_usage.get(key) or 0) + int(
                    planning_usage.get(key) or 0
                )

        await finalize_agent_run(
            session,
            agent_run,
            status=run_status,
            final_answer=final_state.get("final_answer"),
            citations=citations,
            token_usage=token_usage,
            latency_ms=latency_ms,
            nodes_executed=final_state.get("nodes_executed") or [],
        )
        await session.commit()

        return {
            "answer": final_state.get("final_answer") or "",
            "citations": citations,
            "trace_id": trace_id,
            "agent_run_id": str(agent_run.id),
            "nodes_executed": final_state.get("nodes_executed") or [],
            "token_usage": token_usage,
            "status": run_status,
            "plan_id": (
                final_state["execution_plan"].plan_id
                if final_state.get("execution_plan")
                else None
            ),
            "plan_status": final_state.get("plan_status"),
            "planning_source": final_state.get("planning_source"),
            "planning_confidence": final_state.get("planning_confidence"),
            "plan_summary": final_state.get("plan_summary") or [],
            "resolved_query": final_state.get("resolved_query") or query,
            "memory_used": memory_used,
            "memory_written": final_state.get("memory_written") or [],
            "context_budget": final_state.get("context_budget") or {},
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        await finalize_agent_run(
            session,
            agent_run,
            status="failed",
            final_answer=None,
            citations=[],
            token_usage={},
            latency_ms=latency_ms,
            nodes_executed=["failed"],
            error_message=str(exc),
        )
        await session.commit()
        raise
