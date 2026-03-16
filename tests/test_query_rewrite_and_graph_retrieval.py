from __future__ import annotations

from memory.models import MemoryType
from memory.scoring import MemoryScoreWeights
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel
from retrieval.hybrid import HybridRetriever
from retrieval.query_rewrite import QueryRewriter


def test_query_rewriter_expands_short_forms() -> None:
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("How does RAG help LLM memory?")
    assert "retrieval augmented generation" in rewritten.lower()
    assert "large language model" in rewritten.lower()


def test_graph_boost_promotes_connected_memory(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    embedder = SimpleEmbeddingModel(dimensions=32)

    target_text = "Retrieval augmented generation supports long-context question answering."
    other_text = "General productivity notes for grocery planning."

    target_id = store.add_memory(
        MemoryType.SEMANTIC,
        target_text,
        importance_score=0.6,
        embedding=embedder.embed_text(target_text),
    )
    other_id = store.add_memory(
        MemoryType.SEMANTIC,
        other_text,
        importance_score=0.9,
        embedding=embedder.embed_text(other_text),
    )

    store.add_graph_edge(target_id, other_id, relation="supports", weight=3.0)
    store.add_graph_edge(target_id, other_id, relation="related_to", weight=2.0)

    retriever = HybridRetriever(
        store=store,
        embedding_model=embedder,
        query_rewriter=QueryRewriter(),
        score_weights=MemoryScoreWeights(alpha=0.7, beta=0.1, gamma=0.2),
        graph_boost=0.4,
    )

    hits = retriever.retrieve("RAG for long context", limit=2)
    assert hits
    assert "retrieval augmented generation" in hits[0].content.lower()

