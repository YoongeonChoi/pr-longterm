from __future__ import annotations

from memory.manager import MemoryManager
from memory.models import MemoryType
from memory.store import MemoryStore


def test_memory_consolidation_generates_summaries(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    manager = MemoryManager(store)

    store.add_memory(MemoryType.EPISODIC, "User asked about architecture.")
    store.add_memory(MemoryType.EPISODIC, "Agent proposed hierarchical memory.")
    store.add_memory(MemoryType.EPISODIC, "User requested autonomous execution.")

    consolidated = manager.consolidate(max_events=3)
    assert consolidated >= 1

    summaries = store.list_summaries(limit=5)
    assert summaries
    assert "architecture" in summaries[0].summary.lower() or "memory" in summaries[0].summary.lower()

