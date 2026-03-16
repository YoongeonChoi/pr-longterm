from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memory.manager import MemoryManager
from memory.models import MemoryType
from memory.store import MemoryStore
from retrieval.embedding import cosine_similarity


@dataclass(slots=True)
class ConsolidationStats:
    summarized: int
    merged: int
    pruned: int


class MemoryConsolidation:
    """Lifecycle manager: write -> summarize -> merge -> prune(delete)."""

    def __init__(self, store: MemoryStore, manager: MemoryManager) -> None:
        self.store = store
        self.manager = manager

    def summarize_old_memory(self, max_events: int = 100, chunk_size: int = 10) -> int:
        return self.manager.consolidate(max_events=max_events, chunk_size=chunk_size, create_semantic=True)

    def merge_redundant_memory(self, similarity_threshold: float = 0.95, limit: int = 200) -> int:
        semantic_items = self.store.list_memory(memory_type=MemoryType.SEMANTIC, limit=limit)
        merged = 0
        seen: set[str] = set()
        for i, anchor in enumerate(semantic_items):
            if anchor.memory_id in seen or not anchor.embedding:
                continue
            for candidate in semantic_items[i + 1 :]:
                if candidate.memory_id in seen or not candidate.embedding:
                    continue
                sim = cosine_similarity(anchor.embedding, candidate.embedding)
                if sim < similarity_threshold:
                    continue
                merged_text = f"{anchor.content}\n{candidate.content}"
                merged_id = self.store.add_memory(
                    memory_type=MemoryType.SEMANTIC,
                    content=merged_text,
                    importance_score=max(anchor.importance_score, candidate.importance_score),
                    semantic_tags=["merged", "consolidated"],
                    metadata={"source_ids": f"{anchor.memory_id},{candidate.memory_id}"},
                    embedding=anchor.embedding,
                )
                self.store.add_graph_edge(merged_id, anchor.memory_id, "merged_from", weight=1.0)
                self.store.add_graph_edge(merged_id, candidate.memory_id, "merged_from", weight=1.0)
                seen.add(candidate.memory_id)
                merged += 1
                break
        return merged

    def prune_low_importance_memory(
        self,
        min_importance: float = 0.2,
        older_than_days: int = 14,
        limit: int = 500,
    ) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, older_than_days))
        candidates = self.store.list_older_than(cutoff=cutoff, memory_type=None, limit=limit)
        pruned = 0
        for item in candidates:
            if item.importance_score >= min_importance:
                continue
            if self.store.delete_memory(item.memory_id):
                pruned += 1
        return pruned

    def run_full_consolidation(self) -> ConsolidationStats:
        summarized = self.summarize_old_memory(max_events=120, chunk_size=12)
        merged = self.merge_redundant_memory(similarity_threshold=0.96, limit=250)
        pruned = self.prune_low_importance_memory(min_importance=0.18, older_than_days=21, limit=800)
        return ConsolidationStats(summarized=summarized, merged=merged, pruned=pruned)

