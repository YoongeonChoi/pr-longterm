from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.models import MemoryRecord, MemoryType
from memory.store import MemoryStore


@dataclass(slots=True)
class EpisodicMemory:
    """Stores interaction events and session traces."""

    store: MemoryStore

    def write_event(
        self,
        session_id: str,
        user_input: str,
        agent_output: str,
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event_metadata = {"session_id": session_id, "kind": "episode", **(metadata or {})}
        content = f"USER: {user_input}\nASSISTANT: {agent_output}"
        return self.store.add_memory(
            memory_type=MemoryType.EPISODIC,
            content=content,
            importance_score=importance,
            semantic_tags=["conversation", "episode"],
            metadata=event_metadata,
        )

    def read_session_events(self, session_id: str, limit: int = 50) -> list[MemoryRecord]:
        return self.store.search_by_metadata("session_id", session_id, limit=limit)

