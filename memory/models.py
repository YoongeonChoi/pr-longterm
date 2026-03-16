from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVAL = "archival"


def parse_memory_type(value: MemoryType | str) -> MemoryType:
    if isinstance(value, MemoryType):
        return value
    return MemoryType(value)


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    memory_type: MemoryType
    content: str
    importance_score: float
    recency_score: float
    semantic_tags: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass(slots=True)
class MemorySummary:
    summary_id: str
    memory_id: str
    summary: str
    compression_ratio: float
    created_at: datetime

