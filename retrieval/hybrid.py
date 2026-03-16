from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Iterable

from memory.models import MemoryType
from memory.scoring import MemoryScoreWeights, ScoreComponents, compute_memory_score, recency_decay
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel, cosine_similarity
from retrieval.types import RetrievedMemory


def _keyword_overlap_score(query: str, content: str) -> float:
    query_terms = {token.strip().lower() for token in query.split() if token.strip()}
    if not query_terms:
        return 0.0
    content_lower = content.lower()
    matches = sum(1 for term in query_terms if term in content_lower)
    return matches / len(query_terms)


class HybridRetriever:
    def __init__(
        self,
        store: MemoryStore,
        embedding_model: SimpleEmbeddingModel | None = None,
        score_weights: MemoryScoreWeights | None = None,
        recency_half_life_hours: float = 72.0,
        query_rewriter: object | None = None,
        graph_boost: float = 0.0,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self.score_weights = score_weights or MemoryScoreWeights(alpha=0.6, beta=0.2, gamma=0.2)
        self.recency_half_life_hours = recency_half_life_hours
        self.query_rewriter = query_rewriter
        self.graph_boost = max(0.0, graph_boost)

    def _rewrite_query(self, query: str) -> str:
        if self.query_rewriter is None:
            return query
        rewrite = getattr(self.query_rewriter, "rewrite", None)
        if callable(rewrite):
            return str(rewrite(query))
        if callable(self.query_rewriter):
            return str(self.query_rewriter(query))
        return query

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_types: Iterable[MemoryType | str] | None = None,
        candidate_pool: int = 200,
    ) -> list[RetrievedMemory]:
        now = datetime.now(timezone.utc)
        rewritten_query = self._rewrite_query(query)
        query_embedding = self.embedding_model.embed_text(rewritten_query)
        candidates = self.store.list_memory_by_types(memory_types=memory_types, limit=max(limit, candidate_pool))
        scored: list[RetrievedMemory] = []
        graph_scores: dict[str, float] = {}

        if self.graph_boost > 0:
            graph_scores = {candidate.memory_id: self.store.graph_weight_sum(candidate.memory_id) for candidate in candidates}
        max_graph_score = max(graph_scores.values(), default=0.0)

        for candidate in candidates:
            candidate_embedding = candidate.embedding or self.embedding_model.embed_text(candidate.content)
            vector_score = (cosine_similarity(query_embedding, candidate_embedding) + 1.0) / 2.0
            keyword_score = _keyword_overlap_score(rewritten_query, candidate.content)
            relevance = 0.7 * vector_score + 0.3 * keyword_score
            recency = recency_decay(candidate.timestamp, now, half_life_hours=self.recency_half_life_hours)
            importance = candidate.importance_score
            base_score = compute_memory_score(
                components=ScoreComponents(relevance=relevance, recency=recency, importance=importance),
                weights=self.score_weights,
            )
            graph_norm = 0.0
            if max_graph_score > 0:
                graph_norm = graph_scores.get(candidate.memory_id, 0.0) / max_graph_score
            final_score = min(1.0, base_score + self.graph_boost * graph_norm)
            scored.append(
                RetrievedMemory(
                    memory_id=candidate.memory_id,
                    content=candidate.content,
                    memory_type=candidate.memory_type.value,
                    relevance=relevance,
                    recency=recency,
                    importance=importance,
                    final_score=final_score,
                    timestamp=candidate.timestamp,
                    semantic_tags=candidate.semantic_tags,
                    metadata=candidate.metadata,
                    keyword_score=keyword_score,
                )
            )

        scored.sort(key=lambda item: (item.final_score, item.timestamp), reverse=True)
        return scored[: max(1, limit)]

