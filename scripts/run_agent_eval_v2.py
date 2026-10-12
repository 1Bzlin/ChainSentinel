#!/usr/bin/env python3
"""Run the 150-case SupplyGuard Agent Plan Benchmark V2."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.agents.answer_reviewer import review_answer  # noqa: E402
from packages.agents.evidence_types import EvidenceItem  # noqa: E402
from packages.agents.execution import task_executor  # noqa: E402
from packages.agents.graph import run_chat_workflow  # noqa: E402
from packages.agents.planning.entity_extractor import extract_query_entities  # noqa: E402
from packages.agents.planning.plan_validator import validate_execution_plan  # noqa: E402
from packages.agents.planning.schemas import ExecutionPlan  # noqa: E402
from packages.benchmarks.benchmark_runner import load_user_context_for_question  # noqa: E402
from packages.benchmarks.dataset import EvalQuestion  # noqa: E402
from packages.benchmarks.evaluators import ROLE_EMAIL  # noqa: E402
from packages.common.db.models.agent_execution_plan import (  # noqa: E402
    AgentExecutionPlan,
    AgentTaskRun,
)
from packages.common.db.models.document import Document  # noqa: E402
from packages.common.db.models.retrieval_log import RetrievalLog  # noqa: E402
from packages.common.db.session import get_session_factory  # noqa: E402
from packages.mcp_servers.common.schemas import ToolExecutionResult  # noqa: E402
from packages.memory.memory_types import SessionMessage  # noqa: E402
from packages.memory.short_term import append_message, clear_session, now_iso  # noqa: E402
from packages.rag.prompts import NO_EVIDENCE_ANSWER  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "eval" / "supplyguard_agent_eval_v2.jsonl"
DEFAULT_REPORT_DIR = ROOT / "data" / "eval" / "agent_eval_v2_reports"
TASK_TYPES = {"rag", "graph", "tool", "memory"}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    hard: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--dimension", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--keep-sessions", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                cases.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return cases


def dimension_for(case_id: str) -> str:
    prefix = case_id.split("_", 1)[0]
    return {
        "plan": "planner",
        "graph": "planner",
        "mix": "orchestration",
        "sec": "permission",
        "mem": "context_memory",
        "ans": "answer_faithfulness",
        "val": "reliability_audit",
    }.get(prefix, "unknown")


def case_query(case: dict[str, Any]) -> str:
    turns = case.get("turns") or []
    for turn in reversed(turns):
        if turn.get("role") == "user":
            return str(turn.get("content") or "")
    plan = case.get("input_plan") or {}
    if plan.get("original_query"):
        return str(plan["original_query"])
    tasks = plan.get("tasks") or []
    return str(tasks[0].get("query") if tasks else case["id"])


def _add(
    checks: list[Check],
    name: str,
    passed: bool,
    detail: str,
    *,
    hard: bool = False,
) -> None:
    checks.append(Check(name, passed, detail, hard))


def _actual_task_value(task: dict[str, Any], key: str) -> Any:
    tool_input = task.get("tool_input") or {}
    if key == "entity":
        return task.get("entity_code") or task.get("supplier_code") or tool_input.get(
            "supplier_code"
        )
    if key == "supplier_code":
        return task.get("supplier_code") or tool_input.get("supplier_code")
    if key == "tool_input":
        return tool_input
    return task.get(key)


def task_matches(task: dict[str, Any], requirement: dict[str, Any]) -> bool:
    ignored = {"topic", "entity_source"}
    for key, expected in requirement.items():
        if key in ignored:
            continue
        actual = _actual_task_value(task, key)
        if key == "tool_input":
            if not all(actual.get(k) == v for k, v in expected.items()):
                return False
        elif isinstance(expected, str) and isinstance(actual, str):
            if actual.upper() != expected.upper():
                return False
        elif actual != expected:
            return False
    return True


def capability_matches(task: dict[str, Any], capability: str) -> bool:
    return capability in {
        task.get("task_type"),
        task.get("tool_name"),
        task.get("graph_query_type"),
    }


async def load_artifacts(
    session, result: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    plan_row = (
        await session.execute(
            select(AgentExecutionPlan).where(
                AgentExecutionPlan.agent_run_id == uuid.UUID(result["agent_run_id"])
            )
        )
    ).scalar_one_or_none()
    task_rows = list(
        (
            await session.execute(
                select(AgentTaskRun)
                .where(AgentTaskRun.agent_run_id == uuid.UUID(result["agent_run_id"]))
                .order_by(AgentTaskRun.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    retrieval_rows = list(
        (
            await session.execute(
                select(RetrievalLog).where(
                    RetrievalLog.trace_id == result["trace_id"]
                )
            )
        )
        .scalars()
        .all()
    )
    document_ids = {
        str(item.get("document_id"))
        for row in retrieval_rows
        for item in (row.results or [])
        if item.get("document_id")
    }
    slugs: list[str] = []
    if document_ids:
        documents = list(
            (
                await session.execute(
                    select(Document).where(
                        Document.id.in_([uuid.UUID(item) for item in document_ids])
                    )
                )
            )
            .scalars()
            .all()
        )
        slugs = sorted(
            str((doc.metadata_ or {}).get("slug") or doc.title) for doc in documents
        )
    tasks = [dict(row.task_input or {}) | {"status": row.status} for row in task_rows]
    return dict(plan_row.plan_json or {}) if plan_row else {}, tasks, slugs


def infer_outcome(result: dict[str, Any], tasks: list[dict[str, Any]]) -> str:
    if any(task.get("status") == "approval_required" for task in tasks):
        return "approval_required"
    if result.get("answer") == NO_EVIDENCE_ANSWER:
        return "no_evidence"
    return str(result.get("status") or "failed")


def evaluate_e2e(
    case: dict[str, Any],
    result: dict[str, Any],
    plan_json: dict[str, Any],
    task_runs: list[dict[str, Any]],
    slugs: list[str],
    latency_ms: int,
) -> dict[str, Any]:
    expected = case.get("expected") or {}
    expected_plan = expected.get("plan") or {}
    planned = list(plan_json.get("tasks") or [])
    checks: list[Check] = []
    actual_outcome = infer_outcome(result, task_runs)
    wanted_outcome = str(expected.get("outcome"))
    _add(
        checks,
        "outcome",
        actual_outcome == wanted_outcome,
        f"actual={actual_outcome}; expected={wanted_outcome}",
    )

    required = expected_plan.get("required_tasks") or []
    matched = sum(
        1 for requirement in required if any(task_matches(t, requirement) for t in planned)
    )
    if required:
        _add(
            checks,
            "required_tasks",
            matched == len(required),
            f"matched={matched}/{len(required)}",
        )
    capabilities = expected_plan.get("required_capabilities") or []
    capability_hits = sum(
        1 for item in capabilities if any(capability_matches(t, item) for t in planned)
    )
    if capabilities:
        _add(
            checks,
            "required_capabilities",
            capability_hits == len(capabilities),
            f"matched={capability_hits}/{len(capabilities)}",
        )

    # AgentExecutionPlan.plan_json currently keeps the planner proposal, while
    # task_runs reflects the validator-approved executable plan.  For blocked
    # and clarification outcomes, do not report a security failure merely
    # because a forbidden task existed in the pre-validation snapshot.
    effective_tasks = (
        task_runs
        if result.get("plan_status") in {"blocked", "clarification_required"}
        else planned
    )
    forbidden_types = set(expected_plan.get("forbidden_task_types") or [])
    type_hits = sorted(
        {str(task.get("task_type")) for task in effective_tasks} & forbidden_types
    )
    _add(
        checks,
        "forbidden_task_types",
        not type_hits,
        f"hits={type_hits}",
        hard=True,
    )
    forbidden_tools = set(expected_plan.get("forbidden_tools") or [])
    tool_hits = sorted({str(task.get("tool_name")) for task in planned} & forbidden_tools)
    _add(checks, "forbidden_tools", not tool_hits, f"hits={tool_hits}", hard=True)
    forbidden_graph = set(expected_plan.get("forbidden_graph_query_types") or [])
    graph_hits = sorted(
        {str(task.get("graph_query_type")) for task in planned} & forbidden_graph
    )
    _add(
        checks,
        "forbidden_graph_types",
        not graph_hits,
        f"hits={graph_hits}",
        hard=True,
    )
    if "max_task_count" in expected_plan:
        maximum = int(expected_plan["max_task_count"])
        _add(
            checks,
            "max_task_count",
            len(planned) <= maximum,
            f"actual={len(planned)}; maximum={maximum}",
        )
    if "exact_task_count" in expected_plan:
        exact = int(expected_plan["exact_task_count"])
        _add(
            checks,
            "exact_task_count",
            len(planned) == exact,
            f"actual={len(planned)}; expected={exact}",
        )

    security = expected.get("security") or {}
    forbidden_docs = set(
        security.get("must_not_access_docs")
        or security.get("must_not_cite_docs")
        or []
    )
    leaked_docs = sorted(forbidden_docs.intersection(slugs))
    _add(
        checks,
        "forbidden_document_access",
        not leaked_docs,
        f"leaked={leaked_docs}",
        hard=True,
    )
    forbidden_exec_tools = set(security.get("must_not_execute_tools") or [])
    executed_tools = {
        str(task.get("tool_name"))
        for task in task_runs
        if task.get("task_type") == "tool" and task.get("status") != "pending"
    }
    executed_forbidden = sorted(forbidden_exec_tools.intersection(executed_tools))
    _add(
        checks,
        "forbidden_tool_execution",
        not executed_forbidden,
        f"executed={executed_forbidden}",
        hard=True,
    )
    forbidden_graph_exec = set(security.get("must_not_execute_graph_query_types") or [])
    executed_graph = {
        str(task.get("graph_query_type"))
        for task in task_runs
        if task.get("task_type") == "graph" and task.get("status") != "pending"
    }
    graph_exec_hits = sorted(forbidden_graph_exec.intersection(executed_graph))
    _add(
        checks,
        "forbidden_graph_execution",
        not graph_exec_hits,
        f"executed={graph_exec_hits}",
        hard=True,
    )

    evidence = expected.get("evidence") or {}
    required_docs = set(evidence.get("gold_docs") or [])
    if required_docs:
        missing_docs = sorted(required_docs.difference(slugs))
        _add(
            checks,
            "required_documents",
            not missing_docs,
            f"missing={missing_docs}; retrieved={slugs}",
        )

    answer_rules = expected.get("answer") or {}
    answer = str(result.get("answer") or "")
    forbidden_text = [
        text
        for text in answer_rules.get("forbidden_facts") or []
        if str(text).lower() in answer.lower()
    ]
    _add(
        checks,
        "forbidden_answer_facts",
        not forbidden_text,
        f"hits={forbidden_text}",
        hard=True,
    )
    citation_types = answer_rules.get("required_citation_types") or []
    missing_citations = [
        kind
        for kind in citation_types
        if f"[{('doc' if kind == 'rag' else kind)}:" not in answer.lower()
    ]
    if citation_types:
        _add(
            checks,
            "required_citations",
            not missing_citations,
            f"missing={missing_citations}",
        )

    trace = expected.get("trace") or {}
    nodes = list(result.get("nodes_executed") or [])
    required_nodes = trace.get("required_nodes") or []
    missing_nodes = [node for node in required_nodes if node not in nodes]
    if required_nodes:
        _add(checks, "trace_nodes", not missing_nodes, f"missing={missing_nodes}")
    minimum_logs = trace.get("minimum_task_logs")
    if minimum_logs is not None:
        _add(
            checks,
            "minimum_task_logs",
            len(task_runs) >= int(minimum_logs),
            f"actual={len(task_runs)}; minimum={minimum_logs}",
        )

    passed = all(item.passed for item in checks)
    return {
        "id": case["id"],
        "dimension": dimension_for(case["id"]),
        "level": case.get("level"),
        "question_type": case.get("question_type"),
        "role": case.get("user", {}).get("role"),
        "query": case_query(case),
        "execution_mode": "workflow",
        "passed": passed,
        "hard_failure": any(not item.passed and item.hard for item in checks),
        "status": actual_outcome,
        "plan_status": result.get("plan_status"),
        "planning_source": result.get("planning_source"),
        "planned_tasks": planned,
        "task_runs": task_runs,
        "task_recall": matched / len(required) if required else 1.0,
        "retrieved_slugs": slugs,
        "nodes_executed": nodes,
        "resolved_query": result.get("resolved_query"),
        "answer": answer,
        "trace_id": result.get("trace_id"),
        "agent_run_id": result.get("agent_run_id"),
        "latency_ms": latency_ms,
        "token_usage": result.get("token_usage") or {},
        "checks": [asdict(item) for item in checks],
    }


async def user_context_for(session, case: dict[str, Any]):
    role = str(case.get("user", {}).get("role") or "procurement_manager")
    question = EvalQuestion(
        id=case["id"],
        question=case_query(case),
        user_role=role,
        expected_mode="planner",
        question_type=str(case.get("question_type") or "general"),
        user_email=ROLE_EMAIL.get(role),
    )
    return await load_user_context_for_question(session, question)


async def seed_history(case: dict[str, Any], user_context, session_id: str) -> None:
    turns = case.get("turns") or []
    last_user_index = max(
        (index for index, turn in enumerate(turns) if turn.get("role") == "user"),
        default=-1,
    )
    for index, turn in enumerate(turns):
        if index == last_user_index:
            continue
        content = str(turn.get("content") or "")
        await append_message(
            str(user_context.tenant_id),
            str(user_context.user_id),
            session_id,
            SessionMessage(
                role=str(turn.get("role") or "user"),
                content=content,
                timestamp=now_iso(),
                entities=extract_query_entities(content),
                evidence_refs=[],
            ),
        )


async def install_faults(stack: AsyncExitStack, case: dict[str, Any]) -> None:
    faults = {
        str(item.get("target")): str(item.get("behavior"))
        for item in (case.get("runtime", {}).get("fault_injection") or [])
    }
    if not faults:
        return
    original_tool = task_executor.execute_tool_call
    original_rag = task_executor.retrieve_authorized_chunks
    original_graph = task_executor.execute_graph_query

    async def tool_wrapper(session, user_context, tool_name, tool_input=None, **kwargs):
        behavior = faults.get(tool_name)
        if behavior in {"error", "timeout", "no_evidence"}:
            return ToolExecutionResult(
                tool_name=tool_name,
                status="failed",
                result={},
                tool_call_id=str(uuid.uuid4()),
                permission_decision="allowed",
                error_message=f"injected {behavior}",
            )
        return await original_tool(
            session, user_context, tool_name, tool_input, **kwargs
        )

    async def rag_wrapper(*args, **kwargs):
        behavior = faults.get("rag")
        if behavior == "no_evidence":
            return [], True, 1
        if behavior in {"error", "timeout"}:
            raise RuntimeError(f"injected rag {behavior}")
        return await original_rag(*args, **kwargs)

    async def graph_wrapper(*args, **kwargs):
        query_type = kwargs.get("query_type")
        behavior = faults.get("graph") or faults.get(str(query_type))
        if behavior in {"error", "timeout", "no_evidence"}:
            raise RuntimeError(f"injected graph {behavior}")
        return await original_graph(*args, **kwargs)

    await stack.enter_async_context(
        _async_patch("packages.agents.execution.task_executor.execute_tool_call", tool_wrapper)
    )
    await stack.enter_async_context(
        _async_patch(
            "packages.agents.execution.task_executor.retrieve_authorized_chunks",
            rag_wrapper,
        )
    )
    await stack.enter_async_context(
        _async_patch(
            "packages.agents.execution.task_executor.execute_graph_query", graph_wrapper
        )
    )


class _async_patch:
    def __init__(self, target: str, replacement: Any):
        self.patcher = patch(target, replacement)

    async def __aenter__(self):
        return self.patcher.start()

    async def __aexit__(self, exc_type, exc, traceback):
        self.patcher.stop()


async def run_workflow_case(
    case: dict[str, Any], run_id: str, top_k: int, keep_sessions: bool
) -> dict[str, Any]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        user_context = await user_context_for(session, case)
        session_id = str(case.get("runtime", {}).get("session_id") or f"agent-v2:{run_id}:{case['id']}")
        await clear_session(str(user_context.tenant_id), str(user_context.user_id), session_id)
        await seed_history(case, user_context, session_id)
        started = time.perf_counter()
        try:
            async with AsyncExitStack() as stack:
                await install_faults(stack, case)
                result = await run_chat_workflow(
                    session,
                    user_context,
                    case_query(case),
                    session_id=session_id,
                    top_k=top_k,
                    use_memory=bool(case.get("runtime", {}).get("use_memory", True)),
                    trace_id=f"agent-v2-{run_id}-{case['id']}",
                )
            latency_ms = int((time.perf_counter() - started) * 1000)
            plan_json, task_runs, slugs = await load_artifacts(session, result)
            return evaluate_e2e(
                case, result, plan_json, task_runs, slugs, latency_ms
            )
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            return error_row(case, exc, started, "workflow")
        finally:
            if not keep_sessions:
                await clear_session(
                    str(user_context.tenant_id), str(user_context.user_id), session_id
                )


def normalized_plan_payload(case: dict[str, Any]) -> dict[str, Any]:
    raw = dict(case.get("input_plan") or {})
    if not raw:
        raw = {
            "confidence": 0.9,
            "tasks": list(
                case.get("runtime", {})
                .get("planner_output_override", {})
                .get("tasks", [])
            ),
        }
    if raw.get("task_count"):
        raw["tasks"] = [
            {
                "task_id": f"rag_{index}",
                "task_type": "rag",
                "query": str(raw.get("task_template", {}).get("query") or "测试任务"),
            }
            for index in range(int(raw["task_count"]))
        ]
    raw.pop("task_count", None)
    raw.pop("task_template", None)
    query = str(raw.get("original_query") or case_query(case))
    return {
        "plan_id": str(uuid.uuid4()),
        "original_query": query,
        "resolved_query": query,
        "primary_intent": "benchmark_validator",
        "source": "llm",
        "confidence": float(raw.get("confidence", 0.9)),
        "tasks": raw.get("tasks") or [],
        "entities": raw.get("entities") or extract_query_entities(query),
    }


async def run_validator_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    checks: list[Check] = []
    expected = case.get("expected") or {}
    try:
        plan = ExecutionPlan.model_validate(normalized_plan_payload(case))
    except ValidationError as exc:
        actual = "schema_error"
        text = str(exc)
        _add(checks, "outcome", expected.get("outcome") == actual, f"actual={actual}; expected={expected.get('outcome')}")
        wanted_errors = expected.get("validation_error_contains") or []
        missing = [item for item in wanted_errors if item.lower() not in text.lower()]
        _add(checks, "validation_errors", not missing, f"missing={missing}; actual={text}")
        return component_row(case, checks, actual, started, "schema", error=text)

    session_factory = get_session_factory()
    async with session_factory() as session:
        user_context = await user_context_for(session, case)
        try:
            validated = await validate_execution_plan(
                session,
                user_context,
                plan,
                session_id=str(case.get("runtime", {}).get("session_id") or "validator-session"),
                use_memory=bool(case.get("runtime", {}).get("use_memory", False)),
            )
        except Exception as exc:  # noqa: BLE001
            return error_row(case, exc, started, "validator")
    actual = validated.status
    _add(checks, "outcome", actual == expected.get("outcome"), f"actual={actual}; expected={expected.get('outcome')}")
    error_text = "; ".join(validated.validation_errors)
    wanted_errors = expected.get("validation_error_contains") or []
    missing = [item for item in wanted_errors if item.lower() not in error_text.lower()]
    if wanted_errors:
        _add(checks, "validation_errors", not missing, f"missing={missing}; actual={error_text}")
    execution = expected.get("execution") or {}
    retained = execution.get("retained_capabilities") or []
    for capability in retained:
        _add(checks, f"retained:{capability}", any(capability_matches(task.model_dump(), capability) for task in validated.tasks), "capability must remain")
    removed = execution.get("removed_capabilities") or []
    for capability in removed:
        _add(checks, f"removed:{capability}", not any(capability_matches(task.model_dump(), capability) for task in validated.tasks), "capability must be removed", hard=True)
    removed_ids = set(execution.get("removed_task_ids") or [])
    if removed_ids:
        actual_ids = {task.task_id for task in validated.tasks}
        _add(checks, "removed_task_ids", not removed_ids.intersection(actual_ids), f"remaining={sorted(actual_ids)}")
    normalized = execution.get("normalized_task_fields") or {}
    for path, wanted in normalized.items():
        task_id, field = path.split(".", 1)
        task = next((item for item in validated.tasks if item.task_id == task_id), None)
        actual_value = getattr(task, field, None) if task else None
        _add(checks, f"normalized:{path}", actual_value == wanted, f"actual={actual_value}; expected={wanted}")
    return component_row(
        case,
        checks,
        actual,
        started,
        "validator",
        planned_tasks=[task.model_dump() for task in validated.tasks],
        validation_errors=validated.validation_errors,
    )


def review_evidence(case: dict[str, Any]) -> list[EvidenceItem]:
    fixture = case.get("runtime", {}).get("evidence_fixture") or {}
    items: list[EvidenceItem] = []
    for kind, rows in fixture.items():
        evidence_type = {
            "rag": "rag_chunk",
            "graph": "graph_path",
            "tool": "tool_result",
            "memory": "memory",
        }.get(kind, kind)
        for index, row in enumerate(rows):
            source_id = {
                "rag": "00000000-0000-0000-0000-000000000001",
                "graph": "supplier_impact",
                "tool": "query_supplier_profile",
                "memory": "00000000-0000-0000-0000-000000000002",
            }.get(kind, kind)
            metadata: dict[str, Any] = {}
            if kind == "rag":
                metadata["chunk_index"] = index
            elif kind == "graph":
                metadata["path_index"] = index
            elif kind == "tool":
                metadata.update({"tool_name": source_id, "tool_call_id": f"call-{index + 1}"})
            items.append(
                EvidenceItem(
                    evidence_id=f"evidence-{kind}-{index}",
                    evidence_type=evidence_type,
                    source_id=source_id,
                    source_title=kind,
                    content=json.dumps(row, ensure_ascii=False),
                    citation=None,
                    metadata=metadata,
                )
            )
    if not items:
        items.append(
            EvidenceItem(
                evidence_id="known-chunk",
                evidence_type="rag_chunk",
                source_id="00000000-0000-0000-0000-000000000001",
                source_title="fixture",
                content="known evidence",
                metadata={"chunk_index": 0},
            )
        )
    return items


async def run_review_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    session_factory = get_session_factory()
    async with session_factory() as session:
        user_context = await user_context_for(session, case)
    answer = str(case.get("runtime", {}).get("answer_override") or "")
    passed, errors = review_answer(
        answer=answer,
        evidence=review_evidence(case),
        user_context=user_context,
    )
    actual = "completed" if passed else "failed"
    expected = case.get("expected") or {}
    checks: list[Check] = []
    _add(checks, "outcome", actual == expected.get("outcome"), f"actual={actual}; expected={expected.get('outcome')}")
    wanted_reject = bool((expected.get("review") or {}).get("must_reject"))
    if wanted_reject:
        _add(checks, "review_rejection", not passed, f"passed={passed}; errors={errors}")
    wanted_reasons = (expected.get("review") or {}).get("reason_contains") or []
    error_text = "; ".join(errors).lower()
    missing = [reason for reason in wanted_reasons if str(reason).lower() not in error_text]
    if wanted_reasons:
        _add(checks, "review_reason", not missing, f"missing={missing}; actual={errors}")
    return component_row(case, checks, actual, started, "review", review_errors=errors, answer=answer)


def component_row(
    case: dict[str, Any],
    checks: list[Check],
    status: str,
    started: float,
    mode: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "dimension": dimension_for(case["id"]),
        "level": case.get("level"),
        "question_type": case.get("question_type"),
        "role": case.get("user", {}).get("role"),
        "query": case_query(case),
        "execution_mode": mode,
        "passed": all(item.passed for item in checks),
        "hard_failure": any(not item.passed and item.hard for item in checks),
        "status": status,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "task_recall": 1.0,
        "checks": [asdict(item) for item in checks],
        **extra,
    }


def error_row(case: dict[str, Any], exc: Exception, started: float, mode: str) -> dict[str, Any]:
    return {
        "id": case["id"],
        "dimension": dimension_for(case["id"]),
        "level": case.get("level"),
        "question_type": case.get("question_type"),
        "role": case.get("user", {}).get("role"),
        "query": case_query(case),
        "execution_mode": mode,
        "passed": False,
        "hard_failure": False,
        "status": "runner_error",
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "task_recall": 0.0,
        "error": f"{type(exc).__name__}: {exc}",
        "checks": [],
    }


async def run_case(
    case: dict[str, Any], run_id: str, top_k: int, keep_sessions: bool
) -> dict[str, Any]:
    if case.get("input_plan") or case.get("runtime", {}).get("planner_output_override"):
        return await run_validator_case(case)
    if case.get("level") == "review" and case.get("runtime", {}).get("answer_override"):
        return await run_review_case(case)
    return await run_workflow_case(case, run_id, top_k, keep_sessions)


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dimension"])].append(row)
    by_dimension: dict[str, Any] = {}
    for dimension, items in sorted(grouped.items()):
        passed = sum(bool(item.get("passed")) for item in items)
        by_dimension[dimension] = {
            "count": len(items),
            "passed": passed,
            "failed": len(items) - passed,
            "pass_rate": passed / len(items),
            "hard_failures": sum(bool(item.get("hard_failure")) for item in items),
        }
    check_totals: Counter[str] = Counter()
    check_failures: Counter[str] = Counter()
    for row in rows:
        for check in row.get("checks") or []:
            check_totals[check["name"]] += 1
            if not check["passed"]:
                check_failures[check["name"]] += 1
    count = len(rows)
    passed = sum(bool(row.get("passed")) for row in rows)
    latencies = [int(row.get("latency_ms") or 0) for row in rows]
    return {
        "count": count,
        "passed": passed,
        "failed": count - passed,
        "pass_rate": passed / count if count else 0.0,
        "hard_failures": sum(bool(row.get("hard_failure")) for row in rows),
        "runner_errors": sum(row.get("status") == "runner_error" for row in rows),
        "mean_task_recall": sum(float(row.get("task_recall") or 0) for row in rows) / count if count else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "token_usage_total": sum(int((row.get("token_usage") or {}).get("total_tokens") or 0) for row in rows),
        "by_dimension": by_dimension,
        "check_totals": dict(check_totals),
        "check_failures": dict(check_failures),
        "planning_sources": dict(Counter(str(row.get("planning_source")) for row in rows if row.get("planning_source"))),
    }


def write_reports(report_dir: Path, run_id: str, payload: dict[str, Any]) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{run_id}.json"
    csv_path = report_dir / f"{run_id}.csv"
    md_path = report_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fields = ["id", "dimension", "level", "question_type", "role", "execution_mode", "passed", "hard_failure", "status", "plan_status", "planning_source", "task_recall", "latency_ms", "trace_id", "error"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({key: row.get(key) for key in fields})
    summary = payload["summary"]
    lines = [
        "# SupplyGuard Agent Plan Benchmark V2",
        "",
        f"- Run ID: `{run_id}`",
        f"- Cases: {summary['count']}",
        f"- Passed / Failed: {summary['passed']} / {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Hard security failures: {summary['hard_failures']}",
        f"- Runner errors: {summary['runner_errors']}",
        f"- Mean required-task recall: {summary['mean_task_recall']:.2%}",
        f"- P50 / P95 latency: {summary['p50_latency_ms']} / {summary['p95_latency_ms']} ms",
        f"- Total tokens: {summary['token_usage_total']}",
        "",
        "## Dimension results",
        "",
        "| Dimension | Passed | Failed | Total | Pass rate | Hard failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["by_dimension"].items():
        lines.append(f"| {name} | {item['passed']} | {item['failed']} | {item['count']} | {item['pass_rate']:.2%} | {item['hard_failures']} |")
    lines.extend(["", "## Failed cases", ""])
    for row in payload["results"]:
        if row.get("passed"):
            continue
        failed_checks = [check for check in row.get("checks") or [] if not check["passed"]]
        detail = "; ".join(f"{check['name']}: {check['detail']}" for check in failed_checks)
        lines.append(f"- `{row['id']}` ({row['dimension']}): {row.get('error') or detail}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


async def main() -> int:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
    if args.dimension:
        selected = set(args.dimension)
        cases = [case for case in cases if dimension_for(case["id"]) in selected]
    if args.limit is not None:
        cases = cases[: args.limit]
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        row = await run_case(case, run_id, args.top_k, args.keep_sessions)
        rows.append(row)
        print(f"  {'PASS' if row['passed'] else 'FAIL'} status={row.get('status')} latency={row.get('latency_ms')}ms", flush=True)
    payload = {
        "run_id": run_id,
        "dataset": str(args.dataset),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summarize(rows),
        "results": rows,
    }
    payload["reports"] = write_reports(args.report_dir, run_id, payload)
    print("__AGENT_EVAL_V2_RESULT__")
    print(json.dumps({"run_id": run_id, "summary": payload["summary"], "reports": payload["reports"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
