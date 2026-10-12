"""Entity extraction shared by context resolution and planning."""

from __future__ import annotations

import re

ORDER_PATTERN = re.compile(r"(?<![A-Z0-9])(PO-\d{4}-\d{4})(?![A-Z0-9])", re.IGNORECASE)

ENTITY_PATTERNS = {
    "supplier": re.compile(
        r"(?<![A-Z0-9])(SUP-\d{3}|S\d{3})(?![A-Z0-9])", re.IGNORECASE
    ),
    "product": re.compile(r"(?<![A-Z0-9])(P\d{3})(?![A-Z0-9])", re.IGNORECASE),
    "customer": re.compile(r"(?<![A-Z0-9])(C\d{3})(?![A-Z0-9])", re.IGNORECASE),
    "quality": re.compile(r"(?<![A-Z0-9])(QI-\d{3})(?![A-Z0-9])", re.IGNORECASE),
    "order": ORDER_PATTERN,
}


def normalize_supplier_code(code: str) -> str:
    upper = code.upper()
    if upper.startswith("S") and not upper.startswith("SUP-"):
        digits = upper[1:]
        if digits.isdigit():
            return f"SUP-{digits.zfill(3)}"
    return upper


def extract_query_entities(query: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for kind, pattern in ENTITY_PATTERNS.items():
        match = pattern.search(query)
        if not match:
            continue
        value = match.group(1).upper()
        entities[kind] = normalize_supplier_code(value) if kind == "supplier" else value
    return entities
