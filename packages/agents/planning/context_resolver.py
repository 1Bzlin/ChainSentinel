"""Resolve compact session context before planning and retrieval."""

from __future__ import annotations

import re
from typing import Any

from packages.agents.planning.entity_extractor import extract_query_entities

_SUPPLIER_REFERENCES = re.compile(
    r"(它|该供应商|这个供应商|这家供应商|那个供应商|刚才那家)"
)
_PRODUCT_REFERENCES = re.compile(r"(该产品|这个产品|那个产品|它)")
_FOLLOWUP_ONLY = ("继续", "展开", "详细说说", "为什么", "还有呢", "然后呢")


def resolve_query_context(
    query: str,
    recent_entities: dict[str, str] | None,
) -> tuple[str, dict[str, str], list[str], bool]:
    """Return rewritten query, merged entities, references and unresolved flag."""
    historical = dict(recent_entities or {})
    current = extract_query_entities(query)
    entities = {**historical, **current}
    rewritten = query
    references: list[str] = []
    unresolved = False

    if "supplier" not in current and _SUPPLIER_REFERENCES.search(query):
        supplier = historical.get("supplier")
        if supplier:
            rewritten = _SUPPLIER_REFERENCES.sub(supplier, rewritten)
            references.append("previous_supplier")
        else:
            unresolved = True

    if (
        "product" not in current
        and _PRODUCT_REFERENCES.search(query)
        and not references
    ):
        product = historical.get("product")
        if product:
            rewritten = _PRODUCT_REFERENCES.sub(product, rewritten)
            references.append("previous_product")
        elif "它" in query:
            unresolved = True

    if any(keyword in query for keyword in _FOLLOWUP_ONLY) and not current:
        context_bits = [f"{key}={value}" for key, value in historical.items() if value]
        if context_bits:
            rewritten = f"{query}（当前上下文：{', '.join(context_bits)}）"
            references.append("previous_context")
        else:
            unresolved = True

    return rewritten, entities, references, unresolved


def build_session_context(
    messages: list[Any], recent_entities: dict[str, str]
) -> dict[str, Any]:
    turns = [
        {
            "role": getattr(message, "role", "user"),
            "content": str(getattr(message, "content", ""))[:400],
        }
        for message in messages[-4:]
    ]
    return {"active_entities": dict(recent_entities), "recent_turns": turns}
