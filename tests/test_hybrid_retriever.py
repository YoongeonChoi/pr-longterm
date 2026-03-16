from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory.models import MemoryType
from memory.scoring import MemoryScoreWeights
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel
from retrieval.hybrid import HybridRetriever


def test_hybrid_retrieval_prefers_relevant_memory(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    embedder = SimpleEmbeddingModel(dimensions=32)

    relevant = "The system uses hierarchical memory architecture for long context."
    off_topic = "The lunch menu has ramen and dumplings."

    store.add_memory(
        MemoryType.SEMANTIC,
        relevant,
        importance_score=0.7,
        embedding=embedder.embed_text(relevant),
    )
    store.add_memory(
        MemoryType.SEMANTIC,
        off_topic,
        importance_score=0.95,
        embedding=embedder.embed_text(off_topic),
    )

    retriever = HybridRetriever(
        store=store,
        embedding_model=embedder,
        score_weights=MemoryScoreWeights(alpha=0.6, beta=0.1, gamma=0.3),
    )

    hits = retriever.retrieve("hierarchical memory architecture", limit=2)
    assert hits
    assert "hierarchical memory architecture" in hits[0].content.lower()


def test_hybrid_retrieval_applies_recency_boost(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    embedder = SimpleEmbeddingModel(dimensions=32)
    content = "Persistent memory improves multi-session recall."

    older = datetime.now(timezone.utc) - timedelta(days=20)
    newer = datetime.now(timezone.utc) - timedelta(hours=1)

    store.add_memory(
        MemoryType.EPISODIC,
        content,
        importance_score=0.9,
        embedding=embedder.embed_text(content),
        timestamp=older,
    )
    store.add_memory(
        MemoryType.EPISODIC,
        content,
        importance_score=0.6,
        embedding=embedder.embed_text(content),
        timestamp=newer,
    )

    retriever = HybridRetriever(
        store=store,
        embedding_model=embedder,
        score_weights=MemoryScoreWeights(alpha=0.2, beta=0.6, gamma=0.2),
        recency_half_life_hours=24,
    )

    hits = retriever.retrieve("multi-session recall", limit=2)
    assert len(hits) == 2
    assert hits[0].timestamp > hits[1].timestamp

