"""Dependency-aware execution of RAG, graph, tool, and memory tasks."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from packages.agents.evidence_aggregator import (
    evidence_from_graph_paths,
    evidence_from_rag_chunks,
)
from packages.agents.evidence_types import EvidenceItem
from packages.agents.planning.schemas import (
    ExecutionPlan,
    GraphTask,
    MemoryTask,
    PlanTask,
    RagTask,
    TaskResult,
    ToolTask,
)
from packages.authz.audit import write_audit_log
from packages.authz.rbac import UserContext
from packages.common.db.session import get_session_factory
from packages.graphrag.graph_retriever import execute_graph_query
from packages.mcp_servers.common.registry import (
    execute_tool_call,
    format_tool_evidence_content,
)
from packages.memory.memory_retriever import retrieve_memory_evidence
from packages.observability.repositories.memory_event_repo import write_memory_event
from packages.observability.repositories.execution_plan_repo import (
    mark_task_started,
    persist_task_result,
)
from packages.observability.repositories.retrieval_log_repo import write_retrieval_log
from packages.rag.retriever import load_chunk_contents, retrieve_authorized_chunks


def _tag_evidence(
    items: list[EvidenceItem] | list[dict[str, Any]],
    *,
    plan_id: str,
    task_id: str,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for raw in items:
        data = raw.to_dict() if isinstance(raw, EvidenceItem) else dict(raw)
        metadata = dict(data.get("metadata") or {})
        metadata.update({"plan_id": plan_id, "task_id": task_id})
        data["metadata"] = metadata
        tagged.append(data)
    return tagged


async def _execute_rag(
    session: AsyncSession,
    task: RagTask,
    *,
    plan: ExecutionPlan,
    user_context: UserContext,
    trace_id: str,
    agent_run_id: str | None,
    chunk_size_config: int,
    ip_address: str | None,
) -> TaskResult:
    supplier_id = uuid.UUID(task.supplier_id) if task.supplier_id else None
    chunks, filter_applied, latency_ms = await retrieve_authorized_chunks(
        user_context,
        task.query,
        top_k=task.top_k,
        chunk_size_config=chunk_size_config,
        doc_type=task.doc_type,
        supplier_id=supplier_id,
    )
    chunks = await load_chunk_contents(session, chunks)
    result_payload = [
        {
            "document_id": str(item.document_id),
            "chunk_id": str(item.chunk_id),
            "chunk_index": item.chunk_index,
            "score": item.score,
            "title": item.title,
        }
        for item in chunks
    ]
    await write_retrieval_log(
        session,
        tenant_id=user_context.tenant_id,
        user_id=user_context.user_id,
        trace_id=trace_id,
        query=task.query,
        filter_applied=filter_applied,
        top_k=task.top_k,
        chunk_size_config=chunk_size_config,
        results=result_payload,
        latency_ms=latency_ms,
        result_count=len(chunks),
        agent_run_id=agent_run_id,
    )
    await write_audit_log(
        session,
        tenant_id=user_context.tenant_id,
        user_id=user_context.user_id,
        trace_id=trace_id,
        action="plan_rag_task_completed",
        resource_type="rag",
        resource_id=task.task_id,
        permission_decision="allowed",
        reason=f"retrieved {len(chunks)} authorized chunks",
        ip_address=ip_address,
        metadata={
            "plan_id": plan.plan_id,
            "task_id": task.task_id,
            "query": task.query,
        },
    )
    evidence = _tag_evidence(
        evidence_from_rag_chunks(chunks), plan_id=plan.plan_id, task_id=task.task_id
    )
    return TaskResult(
        task_id=task.task_id,
        task_type="rag",
        status="success" if evidence else "no_evidence",
        evidence=evidence,
        latency_ms=latency_ms,
        metadata={"filter_applied": filter_applied, "result_count": len(chunks)},
    )


async def _execute_graph(
    session: AsyncSession,
    task: GraphTask,
    *,
    plan: ExecutionPlan,
    user_context: UserContext,
    trace_id: str,
    agent_run_id: str | None,
    ip_address: str | None,
) -> TaskResult:
    result = await execute_graph_query(
        user_context,
        query=task.query,
        query_type=task.graph_query_type,
        entity_code=task.entity_code,
        top_k=task.top_k,
        trace_id=trace_id,
        agent_run_id=uuid.UUID(agent_run_id) if agent_run_id else None,
        session=session,
        ip_address=ip_address,
    )
    if result.permission_decision == "denied":
        return TaskResult(
            task_id=task.task_id,
            task_type="graph",
            status="denied",
            error=result.error_message or "graph query denied",
            latency_ms=result.latency_ms,
        )
    evidence = _tag_evidence(
        evidence_from_graph_paths(result.graph_paths),
        plan_id=plan.plan_id,
        task_id=task.task_id,
    )
    return TaskResult(
        task_id=task.task_id,
        task_type="graph",
        status="success" if evidence else "no_evidence",
        evidence=evidence,
        latency_ms=result.latency_ms,
        metadata={
            "query_type": task.graph_query_type,
            "entity_code": task.entity_code,
            "result_count": result.result_count,
        },
    )


async def _execute_tool(
    session: AsyncSession,
    task: ToolTask,
    *,
    plan: ExecutionPlan,
    user_context: UserContext,
    trace_id: str,
    agent_run_id: str | None,
) -> TaskResult:
    execution = await execute_tool_call(
        session,
        user_context,
        task.tool_name,
        task.tool_input,
        trace_id=trace_id,
        agent_run_id=agent_run_id,
    )
    citation = f"[tool:{task.tool_name}#{execution.tool_call_id}]"
    evidence = _tag_evidence(
        [
            {
                "evidence_id": execution.tool_call_id or task.tool_name,
                "evidence_type": "tool_result",
                "source_id": task.tool_name,
                "source_title": f"Tool: {task.tool_name}",
                "content": format_tool_evidence_content(
                    task.tool_name, execution.result
                ),
                "score": 1.0 if execution.status == "success" else 0.5,
                "confidence": 0.95 if execution.status == "success" else 0.6,
                "permission_scope": execution.permission_decision,
                "citation": citation,
                "metadata": {
                    "tool_name": task.tool_name,
                    "tool_call_id": execution.tool_call_id,
                    "status": execution.status,
                    "approval_required": execution.approval_required,
                },
            }
        ],
        plan_id=plan.plan_id,
        task_id=task.task_id,
    )
    status = execution.status
    if status not in {"success", "denied", "approval_required", "failed"}:
        status = "failed"
    return TaskResult(
        task_id=task.task_id,
        task_type="tool",
        status=status,
        evidence=evidence,
        error=execution.error_message,
        latency_ms=execution.latency_ms,
        metadata={"tool_name": task.tool_name, "tool_call_id": execution.tool_call_id},
    )


async def _execute_memory(
    session: AsyncSession,
    task: MemoryTask,
    *,
    plan: ExecutionPlan,
    user_context: UserContext,
    trace_id: str,
    agent_run_id: str | None,
) -> TaskResult:
    started = time.perf_counter()
    items, recent_entities = await retrieve_memory_evidence(
        session,
        user_context,
        task.query,
        session_id=task.session_id or trace_id,
        limit=task.limit,
    )
    for item in items:
        await write_memory_event(
            session,
            tenant_id=user_context.tenant_id,
            user_id=user_context.user_id,
            trace_id=trace_id,
            event_type="read",
            memory_type=str(item.get("metadata", {}).get("scope") or "session"),
            summary=str(item.get("content", ""))[:200],
            agent_run_id=uuid.UUID(agent_run_id) if agent_run_id else None,
            metadata={
                "source_id": item.get("source_id"),
                "citation": item.get("citation"),
                "plan_id": plan.plan_id,
                "task_id": task.task_id,
            },
        )
    latency_ms = int((time.perf_counter() - started) * 1000)
    evidence = _tag_evidence(items, plan_id=plan.plan_id, task_id=task.task_id)
    return TaskResult(
        task_id=task.task_id,
        task_type="memory",
        status="success" if evidence else "no_evidence",
        evidence=evidence,
        latency_ms=latency_ms,
        metadata={"recent_entities": recent_entities, "result_count": len(evidence)},
    )


async def _execute_one(
    task: PlanTask,
    *,
    plan: ExecutionPlan,
    user_context: UserContext,
    trace_id: str,
    agent_run_id: str | None,
    chunk_size_config: int,
    ip_address: str | None,
) -> TaskResult:
    session_factory = get_session_factory()
    started = time.perf_counter()
    try:
        async with session_factory() as session:
            await mark_task_started(session, plan.plan_id, task.task_id)
            await session.commit()
            if isinstance(task, RagTask):
                result = await _execute_rag(
                    session,
                    task,
                    plan=plan,
                    user_context=user_context,
                    trace_id=trace_id,
                    agent_run_id=agent_run_id,
                    chunk_size_config=chunk_size_config,
                    ip_address=ip_address,
                )
            elif isinstance(task, GraphTask):
                result = await _execute_graph(
                    session,
                    task,
                    plan=plan,
                    user_context=user_context,
                    trace_id=trace_id,
                    agent_run_id=agent_run_id,
                    ip_address=ip_address,
                )
            elif isinstance(task, ToolTask):
                result = await _execute_tool(
                    session,
                    task,
                    plan=plan,
                    user_context=user_context,
                    trace_id=trace_id,
                    agent_run_id=agent_run_id,
                )
            else:
                result = await _execute_memory(
                    session,
                    task,
                    plan=plan,
                    user_context=user_context,
                    trace_id=trace_id,
                    agent_run_id=agent_run_id,
                )
            await session.commit()
            return result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return TaskResult(
            task_id=task.task_id,
            task_type=task.task_type,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


async def execute_plan(
    plan: ExecutionPlan,
    *,
    user_context: UserContext,
    trace_id: str,
    agent_run_id: str | None,
    chunk_size_config: int,
    ip_address: str | None,
) -> tuple[ExecutionPlan, list[TaskResult]]:
    pending = {task.task_id: task for task in plan.tasks}
    results: dict[str, TaskResult] = {}
    success_states = {"success", "no_evidence"}

    while pending:
        ready = [
            task
            for task in pending.values()
            if all(dependency in results for dependency in task.depends_on)
        ]
        if not ready:
            for task in pending.values():
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    status="failed",
                    error="unresolvable task dependencies",
                )
            break

        executable: list[PlanTask] = []
        for task in ready:
            failed_dependencies = [
                dep
                for dep in task.depends_on
                if results[dep].status not in success_states
            ]
            if failed_dependencies:
                results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    status="skipped",
                    error=f"dependency failed: {', '.join(failed_dependencies)}",
                )
            else:
                executable.append(task)
            pending.pop(task.task_id, None)

        if executable:
            executions = [
                asyncio.wait_for(
                    _execute_one(
                        task,
                        plan=plan,
                        user_context=user_context,
                        trace_id=trace_id,
                        agent_run_id=agent_run_id,
                        chunk_size_config=chunk_size_config,
                        ip_address=ip_address,
                    ),
                    timeout=task.timeout_seconds,
                )
                for task in executable
            ]
            raw_results = await asyncio.gather(*executions, return_exceptions=True)
            for task, raw_result in zip(executable, raw_results, strict=True):
                if isinstance(raw_result, BaseException):
                    result = TaskResult(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        status="failed",
                        error=f"{type(raw_result).__name__}: {raw_result}",
                    )
                else:
                    result = raw_result
                results[task.task_id] = result

    ordered = [results[task.task_id] for task in plan.tasks]
    session_factory = get_session_factory()
    async with session_factory() as result_session:
        for result in ordered:
            await persist_task_result(result_session, plan.plan_id, result)
        await result_session.commit()
    required_failures = [
        result
        for task, result in zip(plan.tasks, ordered, strict=True)
        if task.required and result.status not in success_states
    ]
    any_evidence = any(result.evidence for result in ordered)
    if required_failures:
        plan.status = "partial" if any_evidence else "failed"
    elif any(result.status not in success_states for result in ordered):
        plan.status = "partial"
    else:
        plan.status = "completed"
    return plan, ordered
