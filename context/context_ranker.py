from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from memory.scoring import MemoryScoreWeights, ScoreComponents, compute_memory_score, recency_decay
from retrieval.types import RetrievedMemory


def _keyword_overlap(query: str, content: str) -> float:
    terms = {term.strip().lower() for term in query.split() if term.strip()}
    if not terms:
        return 0.0
    lowered = content.lower()
    hits = sum(1 for term in terms if term in lowered)
    return hits / len(terms)


@dataclass(slots=True)
class RankedContext:
    memory: RetrievedMemory
    relevance: float
    recency: float
    importance: float
    score: float


class ContextRanker:
    """Ranks context candidates by relevance, recency, and importance."""

    def __init__(self, weights: MemoryScoreWeights | None = None, recency_half_life_hours: float = 72.0) -> None:
        self.weights = weights or MemoryScoreWeights(alpha=0.5, beta=0.3, gamma=0.2)
        self.recency_half_life_hours = recency_half_life_hours

    def score(self, query: str, item: RetrievedMemory, now: datetime | None = None) -> RankedContext:
        now = now or datetime.now(timezone.utc)
        relevance = max(item.relevance, _keyword_overlap(query, item.content))
        recency = recency_decay(item.timestamp, now, half_life_hours=self.recency_half_life_hours)
        importance = item.importance
        score = compute_memory_score(
            ScoreComponents(relevance=relevance, recency=recency, importance=importance),
            self.weights,
        )
        return RankedContext(
            memory=item,
            relevance=relevance,
            recency=recency,
            importance=importance,
            score=score,
        )

    def rank(self, query: str, items: list[RetrievedMemory]) -> list[RankedContext]:
        ranked = [self.score(query, item) for item in items]
        ranked.sort(key=lambda item: (item.score, item.memory.timestamp), reverse=True)
        return ranked

