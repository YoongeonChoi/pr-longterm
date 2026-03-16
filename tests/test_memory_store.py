from __future__ import annotations

from datetime import datetime, timezone

from memory.models import MemoryType
from memory.store import MemoryStore


def test_memory_store_write_and_read(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(str(db_path))

    memory_id = store.add_memory(
        memory_type=MemoryType.EPISODIC,
        content="User asked about long-context reasoning.",
        importance_score=0.85,
        semantic_tags=["user", "context"],
        embedding=[0.1, 0.2, 0.3],
        metadata={"session_id": "s-1"},
        timestamp=datetime.now(timezone.utc),
    )

    record = store.get_memory(memory_id)
    assert record is not None
    assert record.memory_id == memory_id
    assert record.memory_type == MemoryType.EPISODIC
    assert record.importance_score == 0.85
    assert "context" in record.semantic_tags
    assert record.metadata["session_id"] == "s-1"


def test_memory_store_keyword_query_returns_relevant_records(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(str(db_path))
    store.add_memory(MemoryType.EPISODIC, "Study notes for memory hierarchy.")
    store.add_memory(MemoryType.SEMANTIC, "Shopping list for home.")

    hits = store.query_keyword("memory hierarchy", limit=5)
    assert hits
    assert "memory hierarchy" in hits[0].content.lower()

