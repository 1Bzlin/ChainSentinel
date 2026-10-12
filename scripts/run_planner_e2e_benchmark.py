#!/usr/bin/env python3
"""Run the planner-v1 workflow against deterministic end-to-end assertions."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.agents.graph import run_chat_workflow  # noqa: E402
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
from packages.memory.memory_types import SessionMessage  # noqa: E402
from packages.memory.short_term import (  # noqa: E402
    append_message,
    clear_session,
    now_iso,
)

DEFAULT_DATASET = ROOT / "data" / "eval" / "planner_e2e_eval.jsonl"
DEFAULT_REPORT_DIR = ROOT / "data" / "eval" / "planner_reports"


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
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--keep-sessions", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return cases


def _task_value(task: dict[str, Any], key: str) -> Any:
    if key == "supplier_code":
        return task.get(key) or (task.get("tool_input") or {}).get(key)
    return task.get(key)


def task_matches(task: dict[str, Any], requirement: dict[str, Any]) -> bool:
    for key, expected in requirement.items():
        if key == "query_contains_all":
            query = str(task.get("query") or "")
            if not all(str(token).lower() in query.lower() for token in expected):
                return False
            continue
        actual = _task_value(task, key)
        if isinstance(expected, str) and isinstance(actual, str):
            if actual.upper() != expected.upper():
                return False
        elif actual != expected:
            return False
    return True


def _check(
    checks: list[Check], name: str, passed: bool, detail: str, *, hard: bool = False
) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail, hard=hard))


async def _load_artifacts(
    session, result: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    plan_row = (
        await session.execute(
            select(AgentExecutionPlan).where(
                AgentExecutionPlan.agent_run_id == uuid.UUID(result["agent_run_id"])
            )
        )
    ).scalar_one()
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
                select(RetrievalLog).where(RetrievalLog.trace_id == result["trace_id"])
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
    slug_by_id: dict[str, str] = {}
    if document_ids:
        docs = list(
            (
                await session.execute(
                    select(Document).where(Document.id.in_([uuid.UUID(i) for i in document_ids]))
                )
            )
            .scalars()
            .all()
        )
        slug_by_id = {
            str(doc.id): str((doc.metadata_ or {}).get("slug") or doc.title)
            for doc in docs
        }
    slugs = sorted(slug_by_id[item_id] for item_id in document_ids if item_id in slug_by_id)
    tasks = [dict(row.task_input or {}) | {"status": row.status} for row in task_rows]
    plan_json = dict(plan_row.plan_json or {})
    return plan_json, tasks, slugs


def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
    plan_json: dict[str, Any],
    task_runs: list[dict[str, Any]],
    retrieved_slugs: list[str],
    latency_ms: int,
) -> dict[str, Any]:
    expected = case.get("expected") or {}
    planned_tasks = list(plan_json.get("tasks") or [])
    checks: list[Check] = []

    statuses = expected.get("statuses")
    if statuses:
        _check(checks, "outcome", result.get("status") in statuses, f"actual={result.get('status')}; expected={statuses}")
    plan_statuses = expected.get("plan_statuses")
    if plan_statuses:
        _check(checks, "plan_status", result.get("plan_status") in plan_statuses, f"actual={result.get('plan_status')}; expected={plan_statuses}")

    requirements = expected.get("required_tasks") or []
    matched = sum(1 for req in requirements if any(task_matches(task, req) for task in planned_tasks))
    task_recall = matched / len(requirements) if requirements else 1.0
    _check(checks, "required_tasks", matched == len(requirements), f"matched={matched}/{len(requirements)}")

    forbidden = expected.get("forbidden_tasks") or []
    forbidden_hits = [req for req in forbidden if any(task_matches(task, req) for task in planned_tasks)]
    _check(checks, "forbidden_tasks", not forbidden_hits, f"hits={forbidden_hits}", hard=True)

    if "max_task_count" in expected:
        maximum = int(expected["max_task_count"])
        _check(checks, "max_task_count", len(planned_tasks) <= maximum, f"actual={len(planned_tasks)}; max={maximum}")
    if "expected_task_run_count" in expected:
        wanted = int(expected["expected_task_run_count"])
        _check(checks, "task_run_count", len(task_runs) == wanted, f"actual={len(task_runs)}; expected={wanted}", hard=wanted == 0)
    if "expected_executed_task_count" in expected:
        wanted = int(expected["expected_executed_task_count"])
        executed = [task for task in task_runs if task.get("status") != "pending"]
        _check(
            checks,
            "executed_task_count",
            len(executed) == wanted,
            f"actual={len(executed)}; expected={wanted}; statuses={[task.get('status') for task in task_runs]}",
            hard=wanted == 0,
        )

    for wanted in expected.get("expected_task_statuses") or []:
        status = wanted.get("status")
        signature = {key: value for key, value in wanted.items() if key != "status"}
        found = any(task_matches(task, signature) and task.get("status") == status for task in task_runs)
        _check(checks, f"task_status:{signature}", found, f"expected_status={status}", hard=status in {"denied", "approval_required"})

    nodes = result.get("nodes_executed") or []
    required_nodes = expected.get("required_nodes") or []
    missing_nodes = [node for node in required_nodes if node not in nodes]
    _check(checks, "required_nodes", not missing_nodes, f"missing={missing_nodes}")

    resolved_query = str(result.get("resolved_query") or "")
    missing_resolved = [token for token in expected.get("resolved_query_contains") or [] if token not in resolved_query]
    _check(checks, "resolved_query", not missing_resolved, f"missing={missing_resolved}")

    forbidden_slugs = set(expected.get("forbidden_retrieved_slugs") or [])
    leaked_slugs = sorted(forbidden_slugs.intersection(retrieved_slugs))
    _check(checks, "forbidden_retrieval", not leaked_slugs, f"leaked={leaked_slugs}", hard=True)
    required_slugs = set(expected.get("required_retrieved_slugs") or [])
    missing_slugs = sorted(required_slugs.difference(retrieved_slugs))
    _check(checks, "required_retrieval", not missing_slugs, f"missing={missing_slugs}")

    answer = str(result.get("answer") or "")
    leaked_text = [text for text in expected.get("answer_not_contains") or [] if text.lower() in answer.lower()]
    _check(checks, "forbidden_answer_text", not leaked_text, f"leaked={leaked_text}", hard=True)

    passed = all(check.passed for check in checks)
    return {
        "id": case["id"],
        "category": case.get("category", "general"),
        "role": case["role"],
        "query": case["query"],
        "passed": passed,
        "hard_failure": any(not check.passed and check.hard for check in checks),
        "status": result.get("status"),
        "plan_status": result.get("plan_status"),
        "planning_source": result.get("planning_source"),
        "planning_confidence": result.get("planning_confidence"),
        "planned_tasks": planned_tasks,
        "task_runs": task_runs,
        "task_recall": task_recall,
        "retrieved_slugs": retrieved_slugs,
        "nodes_executed": nodes,
        "resolved_query": resolved_query,
        "answer": answer,
        "trace_id": result.get("trace_id"),
        "agent_run_id": result.get("agent_run_id"),
        "latency_ms": latency_ms,
        "token_usage": result.get("token_usage") or {},
        "checks": [check.__dict__ for check in checks],
    }


async def run_case(case: dict[str, Any], run_id: str, top_k: int, keep_session: bool) -> dict[str, Any]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        question = EvalQuestion(
            id=case["id"],
            question=case["query"],
            user_role=case["role"],
            expected_mode="planner",
            question_type=case.get("category", "general"),
            user_email=case.get("user_email") or ROLE_EMAIL.get(case["role"]),
        )
        user_context = await load_user_context_for_question(session, question)
        session_id = f"planner-benchmark:{run_id}:{case['id']}"
        await clear_session(str(user_context.tenant_id), str(user_context.user_id), session_id)
        for seed in case.get("session_seed") or []:
            await append_message(
                str(user_context.tenant_id),
                str(user_context.user_id),
                session_id,
                SessionMessage(
                    role=seed.get("role", "user"),
                    content=seed.get("content", ""),
                    timestamp=now_iso(),
                    entities=seed.get("entities") or {},
                    evidence_refs=seed.get("evidence_refs") or [],
                ),
            )
        started = time.perf_counter()
        try:
            result = await run_chat_workflow(
                session,
                user_context,
                case["query"],
                session_id=session_id,
                top_k=top_k,
                use_memory=bool(case.get("use_memory", True)),
                trace_id=f"planner-benchmark-{run_id}-{case['id']}",
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            plan_json, task_runs, slugs = await _load_artifacts(session, result)
            row = evaluate_case(case, result, plan_json, task_runs, slugs, latency_ms)
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": case["id"],
                "category": case.get("category", "general"),
                "role": case["role"],
                "query": case["query"],
                "passed": False,
                "hard_failure": False,
                "status": "runner_error",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
                "checks": [],
            }
            await session.rollback()
        finally:
            if not keep_session:
                await clear_session(str(user_context.tenant_id), str(user_context.user_id), session_id)
        return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    passed = sum(bool(row.get("passed")) for row in rows)
    latencies = sorted(int(row.get("latency_ms") or 0) for row in rows)
    by_category: dict[str, dict[str, int | float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category"))].append(row)
    for category, items in sorted(grouped.items()):
        item_passed = sum(bool(item.get("passed")) for item in items)
        by_category[category] = {
            "count": len(items),
            "passed": item_passed,
            "pass_rate": item_passed / len(items),
        }
    check_counts: Counter[str] = Counter()
    check_failures: Counter[str] = Counter()
    for row in rows:
        for check in row.get("checks") or []:
            check_counts[check["name"]] += 1
            if not check["passed"]:
                check_failures[check["name"]] += 1
    return {
        "count": count,
        "passed": passed,
        "failed": count - passed,
        "pass_rate": passed / count if count else 0.0,
        "hard_failures": sum(bool(row.get("hard_failure")) for row in rows),
        "mean_task_recall": sum(float(row.get("task_recall") or 0) for row in rows) / count if count else 0.0,
        "p50_latency_ms": latencies[(count - 1) // 2] if count else 0,
        "p95_latency_ms": latencies[min(count - 1, max(0, int(count * 0.95)))] if count else 0,
        "token_usage_total": sum(int((row.get("token_usage") or {}).get("total_tokens") or 0) for row in rows),
        "by_category": by_category,
        "check_failures": dict(check_failures),
        "planning_sources": dict(Counter(str(row.get("planning_source")) for row in rows)),
    }


def write_reports(report_dir: Path, run_id: str, payload: dict[str, Any]) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{run_id}.json"
    csv_path = report_dir / f"{run_id}.csv"
    md_path = report_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["id", "category", "role", "passed", "hard_failure", "status", "plan_status", "planning_source", "task_recall", "latency_ms", "trace_id", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({key: row.get(key) for key in fields})
    summary = payload["summary"]
    lines = [
        "# SupplyGuard Planner E2E Benchmark",
        "",
        f"- Run ID: `{run_id}`",
        f"- Cases: {summary['count']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        f"- Hard security failures: {summary['hard_failures']}",
        f"- Mean required-task recall: {summary['mean_task_recall']:.2%}",
        f"- P50/P95 latency: {summary['p50_latency_ms']} / {summary['p95_latency_ms']} ms",
        f"- Total tokens: {summary['token_usage_total']}",
        "",
        "## Category results",
        "",
        "| Category | Passed | Total | Pass rate |",
        "|---|---:|---:|---:|",
    ]
    for category, item in summary["by_category"].items():
        lines.append(f"| {category} | {item['passed']} | {item['count']} | {item['pass_rate']:.2%} |")
    lines.extend(["", "## Failed cases", ""])
    failed_rows = [row for row in payload["results"] if not row.get("passed")]
    if not failed_rows:
        lines.append("None.")
    for row in failed_rows:
        failed_checks = [check for check in row.get("checks") or [] if not check["passed"]]
        detail = "; ".join(f"{item['name']}: {item['detail']}" for item in failed_checks)
        if row.get("error"):
            detail = row["error"]
        lines.append(f"- `{row['id']}`: {detail}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


async def main() -> int:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.category:
        selected = set(args.category)
        cases = [case for case in cases if case.get("category") in selected]
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case.get("id") in selected_ids]
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
        "summary": summarize(rows),
        "results": rows,
    }
    payload["reports"] = write_reports(args.report_dir, run_id, payload)
    print("__PLANNER_BENCHMARK_RESULT__")
    print(json.dumps({"run_id": run_id, "summary": payload["summary"], "reports": payload["reports"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
