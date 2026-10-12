"""Persistence helpers for plans and their task executions."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.agents.planning.schemas import ExecutionPlan, TaskResult
from packages.common.db.models.agent_execution_plan import (
    AgentExecutionPlan,
    AgentTaskRun,
)


async def create_execution_plan(
    session: AsyncSession,
    plan: ExecutionPlan,
    *,
    agent_run_id: str,
    tenant_id: uuid.UUID,
) -> AgentExecutionPlan:
    row = AgentExecutionPlan(
        id=uuid.UUID(plan.plan_id),
        agent_run_id=uuid.UUID(agent_run_id),
        tenant_id=tenant_id,
        source=plan.source,
        confidence=plan.confidence,
        status=plan.status,
        plan_json=plan.model_dump(mode="json"),
        validated_plan_json={},
        model_name=plan.model_name,
        token_usage=plan.token_usage,
    )
    session.add(row)
    await session.flush()
    return row


async def persist_validated_plan(
    session: AsyncSession,
    plan: ExecutionPlan,
    *,
    agent_run_id: str,
    tenant_id: uuid.UUID,
) -> None:
    row = await session.get(AgentExecutionPlan, uuid.UUID(plan.plan_id))
    if row:
        row.status = plan.status
        row.source = plan.source
        row.validated_plan_json = plan.model_dump(mode="json")
    for task in plan.tasks:
        existing = (
            await session.execute(
                select(AgentTaskRun).where(
                    AgentTaskRun.plan_id == uuid.UUID(plan.plan_id),
                    AgentTaskRun.task_id == task.task_id,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            session.add(
                AgentTaskRun(
                    plan_id=uuid.UUID(plan.plan_id),
                    agent_run_id=uuid.UUID(agent_run_id),
                    tenant_id=tenant_id,
                    task_id=task.task_id,
                    task_type=task.task_type,
                    dependencies=task.depends_on,
                    task_input=task.model_dump(mode="json"),
                    status="pending",
                    permission_decision="allowed",
                )
            )
    await session.flush()


async def persist_task_result(
    session: AsyncSession, plan_id: str, result: TaskResult
) -> None:
    row = (
        await session.execute(
            select(AgentTaskRun).where(
                AgentTaskRun.plan_id == uuid.UUID(plan_id),
                AgentTaskRun.task_id == result.task_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return
    row.status = result.status
    row.permission_decision = "denied" if result.status == "denied" else "allowed"
    row.output_summary = {
        "evidence_count": len(result.evidence),
        "metadata": result.metadata,
    }
    row.latency_ms = result.latency_ms
    row.error_message = result.error
    row.completed_at = datetime.datetime.now(datetime.UTC)
    await session.flush()


async def mark_task_started(session: AsyncSession, plan_id: str, task_id: str) -> None:
    row = (
        await session.execute(
            select(AgentTaskRun).where(
                AgentTaskRun.plan_id == uuid.UUID(plan_id),
                AgentTaskRun.task_id == task_id,
            )
        )
    ).scalar_one_or_none()
    if row:
        row.status = "running"
        row.started_at = datetime.datetime.now(datetime.UTC)
        await session.flush()


async def finalize_execution_plan(session: AsyncSession, plan: ExecutionPlan) -> None:
    row = await session.get(AgentExecutionPlan, uuid.UUID(plan.plan_id))
    if row:
        row.status = plan.status
        row.validated_plan_json = plan.model_dump(mode="json")
        row.completed_at = datetime.datetime.now(datetime.UTC)
        await session.flush()
