from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class MemoryScoreWeights:
    alpha: float = 0.6
    beta: float = 0.2
    gamma: float = 0.2

    def normalized(self) -> "MemoryScoreWeights":
        total = self.alpha + self.beta + self.gamma
        if total <= 0:
            return MemoryScoreWeights(alpha=1.0, beta=0.0, gamma=0.0)
        return MemoryScoreWeights(
            alpha=self.alpha / total,
            beta=self.beta / total,
            gamma=self.gamma / total,
        )


@dataclass(slots=True)
class ScoreComponents:
    relevance: float
    recency: float
    importance: float


def recency_decay(timestamp: datetime, current_time: datetime, half_life_hours: float = 24.0) -> float:
    if half_life_hours <= 0:
        return 1.0
    elapsed_seconds = (current_time - timestamp).total_seconds()
    if elapsed_seconds <= 0:
        return 1.0
    elapsed_hours = elapsed_seconds / 3600.0
    return _clamp_01(math.pow(0.5, elapsed_hours / half_life_hours))


def compute_memory_score(components: ScoreComponents, weights: MemoryScoreWeights) -> float:
    w = weights.normalized()
    return (
        _clamp_01(components.relevance) * w.alpha
        + _clamp_01(components.recency) * w.beta
        + _clamp_01(components.importance) * w.gamma
    )

