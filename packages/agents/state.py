"""Agent workflow state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from packages.agents.evidence_types import EvidenceItem
from packages.agents.planning.schemas import ExecutionPlan, TaskResult
from packages.authz.rbac import PermissionDecision, UserContext
from packages.rag.citations import Citation


class AgentState(TypedDict, total=False):
    query: str
    original_query: str
    resolved_query: str
    user_context: UserContext
    trace_id: str
    session_id: str
    agent_run_id: str
    top_k: int
    chunk_size_config: int
    doc_type: str | None
    supplier_id: str | None
    ip_address: str | None
    use_memory: bool
    session_context: dict[str, Any]
    context_references: list[str]
    context_unresolved: bool
    resolved_entities: dict[str, str]
    db_session: Any
    permission_decision: PermissionDecision
    execution_plan: ExecutionPlan
    plan_status: str
    planning_source: str
    planning_confidence: float
    planning_token_usage: dict[str, Any]
    task_results: list[TaskResult]
    plan_summary: list[dict[str, Any]]
    clarification_question: str | None
    recent_entities: dict[str, str]
    unified_evidence: list[EvidenceItem]
    aggregated_evidence: list[dict[str, Any]]
    evidence_summary: dict[str, Any]
    context_budget: dict[str, Any]
    memory_used: list[dict[str, Any]]
    memory_written: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    citations: list[Citation]
    token_usage: dict[str, Any]
    latency_ms: int
    errors: Annotated[list[str], operator.add]
    nodes_executed: Annotated[list[str], operator.add]
    review_passed: bool
    status: str
