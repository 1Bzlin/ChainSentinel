"""Build an execution plan for every authorized request."""

from __future__ import annotations

import uuid
from typing import Any

from packages.agents.planning.llm_planner import generate_plan
from packages.agents.planning.entity_extractor import (
    extract_query_entities,
    normalize_supplier_code,
)
from packages.agents.planning.rule_planner import assess_rules, can_build_rule_plan
from packages.agents.planning.schemas import ExecutionPlan, GraphTask, RagTask, ToolTask


def _normalize_entity(value: str) -> str:
    upper = value.upper()
    return normalize_supplier_code(upper) if upper.startswith(("SUP-", "S")) else upper


def _assert_no_invented_entities(proposal, trusted_entities: dict[str, str]) -> None:
    allowed = {_normalize_entity(value) for value in trusted_entities.values()}
    for task in proposal.tasks:
        values: list[str] = []
        if isinstance(task, RagTask) and task.supplier_code:
            values.append(task.supplier_code)
        elif isinstance(task, GraphTask):
            values.append(task.entity_code)
        elif isinstance(task, ToolTask):
            values.extend(
                str(task.tool_input[key])
                for key in ("supplier_code", "order_no")
                if task.tool_input.get(key)
            )
        for value in values:
            if _normalize_entity(value) not in allowed:
                raise ValueError(f"planner invented entity: {value}")


async def build_execution_plan(
    *,
    original_query: str,
    resolved_query: str,
    session_context: dict[str, Any],
    resolved_entities: dict[str, str],
    context_references: list[str],
    context_unresolved: bool,
    top_k: int,
    doc_type: str | None,
    session_id: str,
    use_memory: bool,
) -> ExecutionPlan:
    assessment = assess_rules(
        resolved_query,
        context_unresolved=context_unresolved,
        top_k=top_k,
        doc_type=doc_type,
        session_id=session_id,
    )
    plan_id = str(uuid.uuid4())
    merged_rule_entities = {**resolved_entities, **assessment.entities}
    if can_build_rule_plan(assessment):
        return ExecutionPlan(
            plan_id=plan_id,
            original_query=original_query,
            resolved_query=resolved_query,
            primary_intent=(
                assessment.signals[0] if assessment.signals else "single_task"
            ),
            source="rule",
            confidence=assessment.confidence,
            tasks=assessment.tasks,
            entities=merged_rule_entities,
            reasoning_tags=[*assessment.signals, *context_references],
        )

    try:
        proposal, usage, model = await generate_plan(
            original_query=original_query,
            resolved_query=resolved_query,
            session_context=session_context,
            resolved_entities=merged_rule_entities,
            assessment=assessment,
            top_k=top_k,
            doc_type=doc_type,
            session_id=session_id,
            use_memory=use_memory,
        )
        trusted_entities = {
            **merged_rule_entities,
            **extract_query_entities(resolved_query),
        }
        _assert_no_invented_entities(proposal, trusted_entities)
        needs_clarification = proposal.needs_clarification or not proposal.tasks
        return ExecutionPlan(
            plan_id=plan_id,
            original_query=original_query,
            resolved_query=proposal.rewritten_query or resolved_query,
            primary_intent=proposal.primary_intent,
            source="llm",
            confidence=proposal.confidence,
            status="clarification_required" if needs_clarification else "ready",
            tasks=[] if needs_clarification else proposal.tasks,
            entities=trusted_entities,
            clarification_question=proposal.clarification_question,
            reasoning_tags=[
                *assessment.signals,
                *context_references,
                *proposal.reasoning_tags,
            ],
            model_name=model,
            token_usage=usage,
        )
    except Exception as exc:
        if context_unresolved or assessment.is_multi_intent or assessment.used_default:
            return ExecutionPlan(
                plan_id=plan_id,
                original_query=original_query,
                resolved_query=resolved_query,
                primary_intent="clarification_required",
                source="fallback",
                confidence=0.4,
                status="clarification_required",
                tasks=[],
                entities=merged_rule_entities,
                clarification_question="请明确希望查询的对象和需要完成的分析内容。",
                reasoning_tags=[
                    *assessment.signals,
                    f"planner_failed:{type(exc).__name__}",
                ],
            )
        return ExecutionPlan(
            plan_id=plan_id,
            original_query=original_query,
            resolved_query=resolved_query,
            primary_intent="rule_fallback",
            source="fallback",
            confidence=max(assessment.confidence, 0.6),
            tasks=assessment.tasks[:1],
            entities=merged_rule_entities,
            reasoning_tags=[
                *assessment.signals,
                f"planner_failed:{type(exc).__name__}",
            ],
        )
