from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class RetrievedMemory:
    memory_id: str
    content: str
    memory_type: str
    relevance: float
    recency: float
    importance: float
    final_score: float
    timestamp: datetime
    semantic_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    keyword_score: float = 0.0

