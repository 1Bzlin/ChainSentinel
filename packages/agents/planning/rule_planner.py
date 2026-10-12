"""Deterministic capability detection and single-task plan fast path."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from packages.agents.planning.entity_extractor import (
    ENTITY_PATTERNS,
    ORDER_PATTERN,
    extract_query_entities,
    normalize_supplier_code,
)
from packages.agents.planning.schemas import (
    GraphTask,
    MemoryTask,
    PlanTask,
    RagTask,
    ToolTask,
)
from packages.memory.memory_policy import is_memory_followup_query

MEMORY_KEYWORDS = ("上次", "之前", "上一轮", "为什么风险", "历史分析", "还记得")
RAG_KEYWORDS = (
    "合同",
    "条款",
    "付款",
    "违约",
    "政策",
    "质检报告内容",
    "报告内容",
    "pricing",
    "payment",
)
NEGATIONS = ("不要", "无需", "不需要", "别查", "不要查询", "仅根据", "只根据")
MULTI_INTENT = (
    "并且",
    "同时",
    "然后",
    "最后",
    "再导出",
    "并导出",
    "分别",
    "以及",
    "还要",
)

TOOL_INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("query_supplier_profile", ("供应商档案", "supplier profile", "供应商基本信息")),
    ("query_supplier_risk_events", ("风险事件", "risk events", "供应商风险")),
    ("query_supplier_delivery_history", ("交付历史", "delivery history", "供货历史")),
    ("query_supplier_contract_summary", ("合同摘要", "contract summary", "合同概要")),
    ("query_delivery_status", ("物流状态", "delivery status", "发货状态", "订单状态")),
    ("query_delay_events", ("延迟事件", "delay events", "物流延迟")),
    ("query_route_info", ("路线信息", "route info", "运输路线")),
    ("export_risk_report", ("导出风险报告", "export risk report", "导出报告")),
    ("export_benchmark_results", ("导出benchmark", "export benchmark", "导出评测")),
]


class RulePlanAssessment(BaseModel):
    tasks: list[PlanTask]
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    entities: dict[str, str] = Field(default_factory=dict)
    has_unresolved_reference: bool = False
    has_negation: bool = False
    is_multi_intent: bool = False
    used_default: bool = False


def _tool_input(query: str, entities: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if entities.get("supplier"):
        result["supplier_code"] = entities["supplier"]
    order_match = ORDER_PATTERN.search(query)
    if order_match:
        result["order_no"] = order_match.group(1).upper()
    return result


def _graph_specs(query: str, entities: dict[str, str]) -> list[tuple[str, str | None]]:
    specs: list[tuple[str, str | None]] = []
    lowered = query.lower()
    if any(token in lowered for token in ("替代供应商", "alternative")):
        specs.append(("supplier_alternatives", entities.get("supplier")))
    if any(token in lowered for token in ("交付延迟", "delay impact", "延迟影响")):
        specs.append(("delivery_delay_impact", entities.get("supplier")))
    if any(token in lowered for token in ("质量问题", "quality issue", "质量追溯")):
        specs.append(("quality_issue_trace", entities.get("quality")))
    if any(
        token in lowered
        for token in ("客户订单依赖", "客户依赖", "customer dependency")
    ):
        specs.append(("customer_order_dependency", entities.get("customer")))
    if any(
        token in lowered
        for token in ("依赖哪些供应商", "产品依赖", "product dependency")
    ):
        specs.append(("product_dependency", entities.get("product")))
    if any(
        token in lowered
        for token in ("影响哪些产品", "影响哪些订单", "供应链路径", "多跳", "图谱")
    ):
        specs.append(("supplier_impact", entities.get("supplier")))
    return list(dict.fromkeys(specs))


def assess_rules(
    query: str,
    *,
    context_unresolved: bool = False,
    top_k: int = 5,
    doc_type: str | None = None,
    session_id: str | None = None,
) -> RulePlanAssessment:
    entities = extract_query_entities(query)
    tasks: list[PlanTask] = []
    signals: list[str] = []
    lowered = query.lower()

    for tool_name, keywords in TOOL_INTENT_PATTERNS:
        if any(keyword.lower() in lowered for keyword in keywords):
            tasks.append(
                ToolTask(
                    task_id=f"tool_{len(tasks) + 1}",
                    query=query,
                    tool_name=tool_name,
                    tool_input=_tool_input(query, entities),
                )
            )
            signals.append(f"tool:{tool_name}")

    for query_type, entity_code in _graph_specs(query, entities):
        if entity_code:
            tasks.append(
                GraphTask(
                    task_id=f"graph_{len(tasks) + 1}",
                    query=query,
                    graph_query_type=query_type,
                    entity_code=entity_code,
                    top_k=min(max(top_k, 1), 20),
                )
            )
            signals.append(f"graph:{query_type}")
        else:
            signals.append(f"graph_missing_entity:{query_type}")

    rag_hits = [keyword for keyword in RAG_KEYWORDS if keyword.lower() in lowered]
    if rag_hits:
        tasks.append(
            RagTask(
                task_id=f"rag_{len(tasks) + 1}",
                query=query,
                top_k=top_k,
                doc_type=doc_type,
                supplier_code=entities.get("supplier"),
            )
        )
        signals.extend(f"rag:{keyword}" for keyword in rag_hits[:3])

    memory_hit = is_memory_followup_query(query) or any(
        k in query for k in MEMORY_KEYWORDS
    )
    if memory_hit:
        tasks.append(
            MemoryTask(
                task_id=f"memory_{len(tasks) + 1}",
                query=query,
                session_id=session_id,
            )
        )
        signals.append("memory:followup")

    used_default = False
    if not tasks:
        tasks.append(
            RagTask(
                task_id="rag_1",
                query=query,
                top_k=top_k,
                doc_type=doc_type,
                supplier_code=entities.get("supplier"),
            )
        )
        signals.append("default:rag_candidate")
        used_default = True

    has_negation = any(token in query for token in NEGATIONS)
    is_multi_intent = len(tasks) > 1 or any(token in query for token in MULTI_INTENT)
    if has_negation:
        signals.append("ambiguity:negation")
    if is_multi_intent:
        signals.append("complex:multi_task")
    if context_unresolved:
        signals.append("ambiguity:unresolved_reference")

    if used_default:
        confidence = 0.45
    elif len(tasks) == 1 and not has_negation and not context_unresolved:
        confidence = 0.95 if isinstance(tasks[0], ToolTask) else 0.88
    else:
        confidence = 0.72
    return RulePlanAssessment(
        tasks=tasks,
        confidence=confidence,
        signals=signals,
        entities=entities,
        has_unresolved_reference=context_unresolved,
        has_negation=has_negation,
        is_multi_intent=is_multi_intent,
        used_default=used_default,
    )


def can_build_rule_plan(assessment: RulePlanAssessment) -> bool:
    if len(assessment.tasks) == 1 and isinstance(assessment.tasks[0], ToolTask):
        task = assessment.tasks[0]
        supplier_tools = {
            "query_supplier_profile",
            "query_supplier_risk_events",
            "query_supplier_delivery_history",
            "query_supplier_contract_summary",
            "query_delay_events",
        }
        if task.tool_name in supplier_tools and not task.tool_input.get(
            "supplier_code"
        ):
            return False
        if task.tool_name == "query_delivery_status" and not (
            task.tool_input.get("supplier_code") or task.tool_input.get("order_no")
        ):
            return False
        if task.tool_name in {"query_route_info", "export_benchmark_results"}:
            return False
    return (
        len(assessment.tasks) == 1
        and assessment.confidence >= 0.85
        and not assessment.has_unresolved_reference
        and not assessment.has_negation
        and not assessment.is_multi_intent
        and not any(
            signal.startswith("graph_missing_entity:") for signal in assessment.signals
        )
    )


def supplier_from_query(query: str) -> str | None:
    match = ENTITY_PATTERNS["supplier"].search(query)
    return normalize_supplier_code(match.group(1)) if match else None
