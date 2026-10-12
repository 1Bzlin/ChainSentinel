"""Report generation node."""

from __future__ import annotations

from packages.agents.citation_manager import build_citation_footer
from packages.agents.evidence_types import EvidenceItem
from packages.agents.state import AgentState
from packages.common.llm.factory import get_llm_client
from packages.rag.citations import Citation
from packages.rag.prompts import (
    NO_EVIDENCE_ANSWER,
    NO_PERMISSION_ANSWER,
    RAG_SYSTEM_PROMPT,
    build_rag_user_prompt,
)


def _evidence_block(items: list[EvidenceItem]) -> str:
    parts: list[str] = []
    for item in items:
        label = item.evidence_type.upper()
        cite = item.citation or ""
        parts.append(f"### {label} {cite}\n{item.content}")
    return "\n\n".join(parts)


def _memory_primary_answer(query: str, items: list[EvidenceItem]) -> str:
    memory_items = [item for item in items if item.evidence_type == "memory"]
    if not memory_items:
        return NO_EVIDENCE_ANSWER
    lines = ["根据历史记忆与上一轮分析：", ""]
    for index, item in enumerate(memory_items[:5], start=1):
        lines.append(f"{index}. {item.content}")
    lines.append("")
    lines.append("记忆证据来源：")
    for item in memory_items[:5]:
        if item.citation:
            lines.append(f"* {item.citation}")
    return "\n".join(lines)


def _tool_primary_answer(query: str, items: list[EvidenceItem]) -> str:
    tool_items = [item for item in items if item.evidence_type == "tool_result"]
    if not tool_items:
        return NO_EVIDENCE_ANSWER
    lines = ["根据工具查询结果：", ""]
    for item in tool_items:
        status = item.metadata.get("status", "success")
        if status == "approval_required":
            lines.append(f"- {item.source_title}: 需要审批后才能执行。")
        elif status == "denied":
            lines.append(f"- {item.source_title}: 权限不足，无法调用。")
        else:
            lines.append(f"- {item.source_title}: {item.content[:500]}")
    lines.append("")
    footer = build_citation_footer(tool_items)
    if footer:
        lines.append(footer)
    return "\n".join(lines)


async def report_node(state: AgentState) -> AgentState:
    if state.get("status") == "blocked":
        return {
            "draft_answer": state.get("draft_answer") or NO_PERMISSION_ANSWER,
            "citations": [],
            "token_usage": {},
            "status": "blocked",
            "nodes_executed": ["report"],
        }

    query = state["query"]
    unified: list[EvidenceItem] = state.get("unified_evidence") or []
    task_results = state.get("task_results") or []
    evidence_types = {item.evidence_type for item in unified}

    if evidence_types == {"tool_result"}:
        answer = _tool_primary_answer(query, unified)
        return {
            "draft_answer": answer,
            "citations": [],
            "token_usage": {},
            "status": "answered",
            "nodes_executed": ["report"],
        }

    if not unified:
        return {
            "draft_answer": NO_EVIDENCE_ANSWER,
            "citations": [],
            "token_usage": {},
            "status": "no_evidence",
            "nodes_executed": ["report"],
        }

    if evidence_types == {"memory"}:
        answer = _memory_primary_answer(query, unified)
        return {
            "draft_answer": answer,
            "citations": [],
            "token_usage": {},
            "status": "answered",
            "nodes_executed": ["report"],
        }

    llm = get_llm_client()
    evidence_block = _evidence_block(unified)
    task_block = "\n".join(
        f"- {result.task_id} ({result.task_type}): {result.status}"
        + (f"; {result.error}" if result.error else "")
        for result in task_results
    )
    user_prompt = build_rag_user_prompt(query, evidence_block)
    if task_block:
        user_prompt = (
            f"执行计划状态：{state.get('plan_status', 'completed')}\n"
            f"子任务执行结果：\n{task_block}\n\n{user_prompt}\n\n"
            "如果部分任务失败、被拒绝、待审批或没有证据，必须明确说明对应部分未完成，"
            "不得用其他证据臆测。"
        )
    chat_result = await llm.chat(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    answer = (chat_result.content or "").strip() or NO_EVIDENCE_ANSWER
    footer = build_citation_footer(unified)
    if footer and "证据来源" not in answer:
        answer = f"{answer}\n\n{footer}"

    citations = [
        Citation(
            document_id=item.source_id,
            chunk_id=item.evidence_id,
            chunk_index=int(item.metadata.get("chunk_index", 0)),
            title=item.source_title,
            score=item.score,
            snippet=item.content[:280],
        )
        for item in unified
        if item.evidence_type == "rag_chunk"
    ]

    return {
        "draft_answer": answer,
        "citations": citations,
        "token_usage": chat_result.token_usage,
        "status": "answered" if state.get("plan_status") != "partial" else "partial",
        "nodes_executed": ["report"],
    }
