from __future__ import annotations

from dataclasses import dataclass

from memory.models import MemorySummary, MemoryType
from memory.store import MemoryStore


@dataclass(slots=True)
class ArchivalMemory:
    """Stores compressed historical records and summary checkpoints."""

    store: MemoryStore

    def archive_snapshot(self, summary: str, source_memory_id: str, compression_ratio: float) -> str:
        return self.store.upsert_summary(
            memory_id=source_memory_id,
            summary=summary,
            compression_ratio=compression_ratio,
        )

    def list_archives(self, limit: int = 50) -> list[MemorySummary]:
        return self.store.list_summaries(limit=limit)

    def write_archival_note(self, content: str, importance: float = 0.6) -> str:
        return self.store.add_memory(
            memory_type=MemoryType.ARCHIVAL,
            content=content,
            importance_score=importance,
            semantic_tags=["archive", "summary"],
            metadata={"kind": "archival_note"},
        )

