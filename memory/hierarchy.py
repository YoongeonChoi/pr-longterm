from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.manager import MemoryManager
from retrieval.hybrid import HybridRetriever
from retrieval.types import RetrievedMemory


@dataclass(slots=True)
class HierarchicalRecall:
    session_id: str
    working_state: dict[str, Any]
    memories: list[RetrievedMemory] = field(default_factory=list)


class HierarchicalMemoryRuntime:
    def __init__(self, memory_manager: MemoryManager, retriever: HybridRetriever) -> None:
        self.memory_manager = memory_manager
        self.retriever = retriever
        self._working_by_session: dict[str, dict[str, Any]] = {}

    def set_working_memory(self, session_id: str, key: str, value: Any) -> None:
        session_state = self._working_by_session.setdefault(session_id, {})
        session_state[key] = value

    def get_working_memory(self, session_id: str) -> dict[str, Any]:
        return dict(self._working_by_session.get(session_id, {}))

    def write_turn(self, session_id: str, user_input: str, agent_output: str, importance: float = 0.75) -> str:
        metadata = {"session_id": session_id}
        memory_id = self.memory_manager.write_interaction(
            user_input=user_input,
            agent_output=agent_output,
            importance=importance,
            semantic_tags=["interaction", "session"],
            metadata=metadata,
            extract_semantic=True,
        )
        self.set_working_memory(session_id, "last_user_input", user_input)
        self.set_working_memory(session_id, "last_agent_output", agent_output)
        return memory_id

    def recall(self, session_id: str, query: str, limit: int = 5) -> HierarchicalRecall:
        memories = self.retriever.retrieve(query=query, limit=max(limit, 20))
        session_hits = [item for item in memories if item.metadata.get("session_id") == session_id]
        if len(session_hits) < limit:
            # Backfill with top global hits when session-local hits are insufficient.
            seen = {item.memory_id for item in session_hits}
            for item in memories:
                if item.memory_id in seen:
                    continue
                session_hits.append(item)
                if len(session_hits) >= limit:
                    break
        return HierarchicalRecall(
            session_id=session_id,
            working_state=self.get_working_memory(session_id),
            memories=session_hits[:limit],
        )

    def consolidate_session(self, session_id: str, max_events: int = 100, chunk_size: int = 10) -> int:
        history = self.memory_manager.read_session_history(session_id=session_id, limit=max_events)
        if not history:
            return 0
        # Consolidation currently runs globally in MemoryManager; this call still helps maintain periodic summaries.
        return self.memory_manager.consolidate(max_events=max_events, chunk_size=chunk_size, create_semantic=True)

