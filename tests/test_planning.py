"""Tests for the all-requests-are-plans workflow."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from packages.agents.planning.context_resolver import resolve_query_context
from packages.agents.planning.llm_planner import parse_plan_response
from packages.agents.planning.plan_builder import build_execution_plan
from packages.agents.planning.plan_validator import (
    _graph_entity_matches,
    _has_cycle,
    _missing_tool_inputs,
)
from packages.agents.planning.rule_planner import assess_rules, can_build_rule_plan
from packages.agents.planning.schemas import GraphTask, RagTask
from packages.agents.planning.schemas import ExecutionPlan, TaskResult, ToolTask
from packages.authz.rbac import UserContext
from packages.agents.context_budget_manager import apply_context_budget
from packages.agents.evidence_types import EvidenceItem
import uuid


def test_resolves_supplier_pronoun_from_session_context() -> None:
    rewritten, entities, references, unresolved = resolve_query_context(
        "它影响哪些订单？", {"supplier": "SUP-001"}
    )
    assert rewritten == "SUP-001影响哪些订单？"
    assert entities["supplier"] == "SUP-001"
    assert references == ["previous_supplier"]
    assert unresolved is False


def test_clear_graph_request_builds_single_task_rule_plan() -> None:
    assessment = assess_rules("SUP-001影响哪些订单？")
    assert len(assessment.tasks) == 1
    assert isinstance(assessment.tasks[0], GraphTask)
    assert assessment.tasks[0].graph_query_type == "supplier_impact"
    assert can_build_rule_plan(assessment) is True


def test_complex_request_detects_multiple_capabilities() -> None:
    assessment = assess_rules("分析SUP-001影响哪些订单，同时查看合同条款和物流状态")
    assert {task.task_type for task in assessment.tasks} == {"graph", "rag", "tool"}
    assert assessment.is_multi_intent is True
    assert can_build_rule_plan(assessment) is False


def test_parses_multi_task_plan() -> None:
    proposal = parse_plan_response("""```json
        {
          "primary_intent": "综合供应商影响",
          "rewritten_query": "分析SUP-001的订单影响和合同责任",
          "tasks": [
            {
              "task_id": "g1", "task_type": "graph",
              "query": "SUP-001影响哪些订单", "depends_on": [],
              "required": true, "timeout_seconds": 30,
              "graph_query_type": "supplier_impact",
              "entity_code": "SUP-001", "top_k": 20
            },
            {
              "task_id": "r1", "task_type": "rag",
              "query": "SUP-001合同延期责任", "depends_on": [],
              "required": true, "timeout_seconds": 30,
              "top_k": 5, "doc_type": "contract",
              "supplier_code": "SUP-001", "supplier_id": null
            }
          ],
          "entities": {"supplier": "SUP-001"},
          "confidence": 0.93,
          "needs_clarification": false,
          "clarification_question": null,
          "reasoning_tags": ["graph_and_document"]
        }
        ```""")
    assert [task.task_type for task in proposal.tasks] == ["graph", "rag"]
    assert proposal.confidence == 0.93


def test_plan_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        parse_plan_response("""{
              "primary_intent":"x", "rewritten_query":"x", "tasks": [],
              "entities":{}, "confidence":0.8, "needs_clarification":true,
              "clarification_question":"请明确对象", "reasoning_tags":[],
              "sql":"select * from suppliers"
            }""")


def test_cycle_detection_rejects_task_dependency_cycle() -> None:
    tasks = [
        RagTask(task_id="a", query="a", depends_on=["b"]),
        RagTask(task_id="b", query="b", depends_on=["a"]),
    ]
    assert _has_cycle(tasks) is True


def test_graph_entity_type_and_tool_inputs_are_validated() -> None:
    assert _graph_entity_matches(
        GraphTask(
            task_id="g1",
            query="product",
            graph_query_type="product_dependency",
            entity_code="P001",
        )
    )
    assert not _graph_entity_matches(
        GraphTask(
            task_id="g2",
            query="product",
            graph_query_type="product_dependency",
            entity_code="PO-2026-0001",
        )
    )
    assert _missing_tool_inputs(
        ToolTask(
            task_id="t1",
            query="route",
            tool_name="query_route_info",
            tool_input={"origin": "Shanghai"},
        )
    ) == ["destination"]


def test_complex_planner_failure_requests_clarification(monkeypatch) -> None:
    async def fail_generation(**kwargs):
        raise RuntimeError("planner unavailable")

    monkeypatch.setattr(
        "packages.agents.planning.plan_builder.generate_plan", fail_generation
    )
    plan = asyncio.run(
        build_execution_plan(
            original_query="分析SUP-001影响哪些订单，同时查看合同条款",
            resolved_query="分析SUP-001影响哪些订单，同时查看合同条款",
            session_context={},
            resolved_entities={"supplier": "SUP-001"},
            context_references=[],
            context_unresolved=False,
            top_k=5,
            doc_type=None,
            session_id="session-1",
            use_memory=True,
        )
    )
    assert plan.status == "clarification_required"
    assert plan.tasks == []
    assert plan.source == "fallback"


def test_plan_executor_runs_dependency_waves(monkeypatch) -> None:
    from packages.agents.execution import task_executor

    events: list[str] = []

    async def fake_execute_one(task, **kwargs):
        events.append(f"start:{task.task_id}")
        await asyncio.sleep(0)
        events.append(f"end:{task.task_id}")
        return TaskResult(
            task_id=task.task_id,
            task_type=task.task_type,
            status="success",
        )

    async def fake_persist(*args, **kwargs):
        return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def commit(self):
            return None

    monkeypatch.setattr(task_executor, "_execute_one", fake_execute_one)
    monkeypatch.setattr(task_executor, "persist_task_result", fake_persist)
    monkeypatch.setattr(
        task_executor, "get_session_factory", lambda: lambda: FakeSession()
    )

    plan = ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        original_query="test",
        resolved_query="test",
        primary_intent="test",
        source="rule",
        confidence=1.0,
        tasks=[
            RagTask(task_id="a", query="a"),
            RagTask(task_id="b", query="b"),
            RagTask(task_id="c", query="c", depends_on=["a", "b"]),
        ],
    )
    user = UserContext(
        tenant_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        email="admin@example.com",
        department="procurement",
        roles=["admin"],
    )
    completed_plan, results = asyncio.run(
        task_executor.execute_plan(
            plan,
            user_context=user,
            trace_id="trace",
            agent_run_id=None,
            chunk_size_config=512,
            ip_address=None,
        )
    )

    assert events.index("start:c") > events.index("end:a")
    assert events.index("start:c") > events.index("end:b")
    assert [result.status for result in results] == ["success", "success", "success"]
    assert completed_plan.status == "completed"


def test_context_budget_keeps_tool_evidence() -> None:
    item = EvidenceItem(
        evidence_id="call-1",
        evidence_type="tool_result",
        source_id="query_supplier_profile",
        source_title="Tool",
        content="supplier profile",
        score=1.0,
        confidence=0.95,
    )
    selected, metadata = apply_context_budget([item], max_context_tokens=100)
    assert selected == [item]
    assert metadata["tool_count"] == 1
