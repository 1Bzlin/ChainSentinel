"""Strict contracts for planning and executing agent work."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BasePlanTask(StrictModel):
    task_id: str = Field(min_length=1, max_length=64)
    task_type: str
    query: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    timeout_seconds: int = Field(default=30, ge=1, le=120)


class RagTask(BasePlanTask):
    task_type: Literal["rag"] = "rag"
    top_k: int = Field(default=5, ge=1, le=20)
    doc_type: str | None = None
    supplier_code: str | None = None
    supplier_id: str | None = None


class GraphTask(BasePlanTask):
    task_type: Literal["graph"] = "graph"
    graph_query_type: str
    entity_code: str
    top_k: int = Field(default=20, ge=1, le=50)


class ToolTask(BasePlanTask):
    task_type: Literal["tool"] = "tool"
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class MemoryTask(BasePlanTask):
    task_type: Literal["memory"] = "memory"
    session_id: str | None = None
    limit: int = Field(default=10, ge=1, le=20)


PlanTask = Annotated[
    RagTask | GraphTask | ToolTask | MemoryTask,
    Field(discriminator="task_type"),
]


class PlanProposal(StrictModel):
    primary_intent: str
    rewritten_query: str
    tasks: list[PlanTask] = Field(default_factory=list, max_length=8)
    entities: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: str | None = None
    reasoning_tags: list[str] = Field(default_factory=list)


class ExecutionPlan(StrictModel):
    plan_id: str
    original_query: str
    resolved_query: str
    primary_intent: str
    source: Literal["rule", "llm", "fallback", "system", "guard"]
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "ready",
        "clarification_required",
        "blocked",
        "completed",
        "partial",
        "failed",
    ] = "ready"
    tasks: list[PlanTask] = Field(default_factory=list, max_length=8)
    entities: dict[str, str] = Field(default_factory=dict)
    clarification_question: str | None = None
    reasoning_tags: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    model_name: str | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)


class TaskResult(StrictModel):
    task_id: str
    task_type: Literal["rag", "graph", "tool", "memory"]
    status: Literal[
        "success",
        "no_evidence",
        "denied",
        "approval_required",
        "failed",
        "skipped",
    ]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def task_summary(task: PlanTask, result: TaskResult | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "query": task.query,
        "depends_on": task.depends_on,
        "required": task.required,
        "status": result.status if result else "pending",
    }
    if isinstance(task, GraphTask):
        summary["graph_query_type"] = task.graph_query_type
        summary["entity_code"] = task.entity_code
    elif isinstance(task, ToolTask):
        summary["tool_name"] = task.tool_name
    elif isinstance(task, RagTask):
        summary["doc_type"] = task.doc_type
        summary["supplier_code"] = task.supplier_code
    if result and result.error:
        summary["error"] = result.error
    return summary
