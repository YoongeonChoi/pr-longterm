from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from agent.context_manager import ContextManager
from compression.strategies import ContextCompressor
from retrieval.types import RetrievedMemory


@dataclass
class StubRetriever:
    entries: list[RetrievedMemory]

    def retrieve(self, query: str, limit: int = 5) -> list[RetrievedMemory]:
        return self.entries[:limit]


def _memory(content: str) -> RetrievedMemory:
    now = datetime.now(timezone.utc)
    return RetrievedMemory(
        memory_id=content[:8],
        content=content,
        memory_type="episodic",
        relevance=0.7,
        recency=0.8,
        importance=0.5,
        final_score=0.7,
        timestamp=now,
        semantic_tags=[],
        metadata={},
    )


def test_context_manager_compresses_when_over_budget() -> None:
    long_memories = [_memory("memory " * 100), _memory("context " * 100)]
    retriever = StubRetriever(entries=long_memories)
    manager = ContextManager(
        token_budget=60,
        retriever=retriever,
        compressor=ContextCompressor(),
    )

    package = manager.build_context(
        query="memory",
        conversation=["user says " + "x " * 30],
        retrieval_limit=2,
    )
    assert package.total_tokens <= 60
    assert package.was_compressed is True
    assert package.memories


def test_context_manager_keeps_full_context_if_within_budget() -> None:
    short_memories = [_memory("short fact about memory"), _memory("another short fact")]
    retriever = StubRetriever(entries=short_memories)
    manager = ContextManager(
        token_budget=200,
        retriever=retriever,
        compressor=ContextCompressor(),
    )

    package = manager.build_context(query="memory", conversation=["hello"], retrieval_limit=2)
    assert package.was_compressed is False
    assert package.total_tokens < 200

