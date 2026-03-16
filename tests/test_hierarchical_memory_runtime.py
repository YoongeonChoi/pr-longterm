from __future__ import annotations

from memory.hierarchy import HierarchicalMemoryRuntime
from memory.manager import MemoryManager
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel
from retrieval.hybrid import HybridRetriever
from retrieval.query_rewrite import QueryRewriter


def test_hierarchical_runtime_tracks_working_and_session_memories(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    manager = MemoryManager(store)
    retriever = HybridRetriever(
        store=store,
        embedding_model=SimpleEmbeddingModel(dimensions=32),
        query_rewriter=QueryRewriter(),
    )
    runtime = HierarchicalMemoryRuntime(memory_manager=manager, retriever=retriever)

    runtime.set_working_memory("s-1", "goal", "build memory agent")
    runtime.write_turn("s-1", "We need long conversation memory.", "Agreed, we store episodic turns.")

    recall = runtime.recall("s-1", "conversation memory", limit=5)
    assert recall.working_state["goal"] == "build memory agent"
    assert recall.memories
    assert any("conversation memory" in item.content.lower() for item in recall.memories)

