"""Unified evidence item model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    # 证据的唯一标识，用于去重、追踪和引用具体证据。
    evidence_id: str
    # 证据类型：rag_chunk、graph_path、tool_result 或 memory。
    evidence_type: str
    # 来源对象的标识，例如文档 ID、图查询模板 ID 或工具名称。
    source_id: str
    # 便于展示和理解的来源标题。
    source_title: str
    # 提供给模型生成答案的证据正文。
    content: str
    # 检索或排序分数，数值越高表示与当前查询越相关。
    score: float = 0.0
    # 对证据可靠程度的估计，通常取值范围为 0 到 1。
    confidence: float = 0.0
    # 证据通过权限校验后的访问范围或权限判定结果。
    permission_scope: str = ""
    # 答案中使用的可追溯引用标记；无引用时为 None。
    citation: str | None = None
    # 不适合放入通用字段的来源专属扩展信息。
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "content": self.content,
            "score": self.score,
            "confidence": self.confidence,
            "permission_scope": self.permission_scope,
            "citation": self.citation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceItem:
        return cls(
            evidence_id=str(data.get("evidence_id", "")),
            evidence_type=str(data.get("evidence_type", "")),
            source_id=str(data.get("source_id", "")),
            source_title=str(data.get("source_title", "")),
            content=str(data.get("content", "")),
            score=float(data.get("score", 0)),
            confidence=float(data.get("confidence", 0)),
            permission_scope=str(data.get("permission_scope", "")),
            citation=data.get("citation"),
            metadata=data.get("metadata") or {},
        )
