from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory.scoring import MemoryScoreWeights, ScoreComponents, compute_memory_score, recency_decay


def test_recency_decay_half_life_behavior() -> None:
    now = datetime.now(timezone.utc)
    one_day_ago = now - timedelta(hours=24)
    score = recency_decay(one_day_ago, now, half_life_hours=24)
    assert 0.49 <= score <= 0.51


def test_memory_score_weighted_sum() -> None:
    components = ScoreComponents(relevance=0.9, recency=0.5, importance=0.2)
    weights = MemoryScoreWeights(alpha=0.5, beta=0.3, gamma=0.2)

    result = compute_memory_score(components, weights)
    expected = 0.9 * 0.5 + 0.5 * 0.3 + 0.2 * 0.2
    assert abs(result - expected) < 1e-8

