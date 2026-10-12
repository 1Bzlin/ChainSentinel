"""Merge Secure RAG, GraphRAG, and Memory evidence."""

from __future__ import annotations

from packages.agents.evidence_aggregator import aggregate_evidence
from packages.agents.evidence_types import EvidenceItem
from packages.agents.state import AgentState
from packages.common.config import get_settings


async def evidence_aggregator_node(state: AgentState) -> AgentState:
    settings = get_settings()
    max_tokens = getattr(settings, "MAX_CONTEXT_TOKENS", 8000)

    planned_evidence = [
        EvidenceItem.from_dict(item)
        for result in (state.get("task_results") or [])
        for item in result.evidence
    ]
    selected, summary = aggregate_evidence(
        evidence_items=planned_evidence,
        max_context_tokens=max_tokens,
    )

    return {
        "unified_evidence": selected,
        "aggregated_evidence": [item.to_dict() for item in selected],
        "evidence_summary": summary,
        "context_budget": {
            "max_tokens": summary.get("max_tokens", max_tokens),
            "used_tokens": summary.get("used_tokens", 0),
        },
        "memory_used": [
            item.to_dict() for item in selected if item.evidence_type == "memory"
        ],
        "nodes_executed": ["evidence_aggregator"],
    }
