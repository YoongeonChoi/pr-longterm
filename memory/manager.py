from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from memory.models import MemoryRecord, MemoryType
from memory.store import KeywordSearchHit, MemoryStore


def _summarize_texts(texts: list[str], max_words: int = 80) -> str:
    merged = " ".join(texts)
    words = merged.split()
    if len(words) <= max_words:
        return merged
    head = words[: max_words // 2]
    tail = words[-(max_words // 2) :]
    return " ".join([*head, "...", *tail])


def _extract_facts(text: str, max_facts: int = 3) -> list[str]:
    sentences = [part.strip() for part in re.split(r"[.!?\n]+", text) if part.strip()]
    facts: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in (" is ", " are ", " means ", " includes ", " use ")):
            facts.append(sentence)
        if len(facts) >= max_facts:
            break
    return facts


@dataclass(slots=True)
class MemoryManager:
    store: MemoryStore
    working_memory: dict[str, Any] = field(default_factory=dict)

    def set_working_state(self, key: str, value: Any) -> None:
        self.working_memory[key] = value

    def get_working_state(self, key: str, default: Any = None) -> Any:
        return self.working_memory.get(key, default)

    def write_interaction(
        self,
        user_input: str,
        agent_output: str,
        importance: float = 0.7,
        semantic_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        extract_semantic: bool = True,
    ) -> str:
        content = f"USER: {user_input}\nASSISTANT: {agent_output}"
        tags = semantic_tags or ["interaction"]
        meta = metadata or {}
        memory_id = self.store.add_memory(
            memory_type=MemoryType.EPISODIC,
            content=content,
            importance_score=importance,
            semantic_tags=tags,
            metadata=meta,
        )
        if extract_semantic:
            self.extract_semantic_from_text(
                text=f"{user_input}. {agent_output}",
                source_memory_id=memory_id,
                metadata=meta,
            )
        return memory_id

    def extract_semantic_from_text(
        self,
        text: str,
        source_memory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        facts = _extract_facts(text)
        added: list[str] = []
        for fact in facts:
            fact_metadata = dict(metadata or {})
            if source_memory_id is not None:
                fact_metadata["source_memory_id"] = source_memory_id
            added.append(
                self.add_semantic_fact(
                    fact=fact,
                    importance=0.75,
                    semantic_tags=["fact", "extracted"],
                    metadata=fact_metadata,
                )
            )
        return added

    def add_semantic_fact(
        self,
        fact: str,
        importance: float = 0.8,
        semantic_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        tags = semantic_tags or ["fact"]
        return self.store.add_memory(
            memory_type=MemoryType.SEMANTIC,
            content=fact,
            importance_score=importance,
            semantic_tags=tags,
            metadata=metadata or {},
        )

    def read_session_history(self, session_id: str, limit: int = 50) -> list[MemoryRecord]:
        if not session_id:
            return []
        return self.store.search_by_metadata("session_id", session_id, limit=limit)

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        memory_types: Iterable[MemoryType | str] | None = None,
    ) -> list[KeywordSearchHit]:
        return self.store.query_keyword(query=query, limit=limit, memory_types=memory_types)

    def consolidate(
        self,
        max_events: int = 50,
        chunk_size: int = 5,
        create_semantic: bool = True,
    ) -> int:
        events = self.store.list_memory(memory_type=MemoryType.EPISODIC, limit=max_events)
        if not events:
            return 0

        summary_count = 0
        for idx in range(0, len(events), max(1, chunk_size)):
            chunk = events[idx : idx + chunk_size]
            texts = [event.content for event in chunk]
            summary = _summarize_texts(texts)
            original_words = sum(max(1, len(text.split())) for text in texts)
            summary_words = max(1, len(summary.split()))
            ratio = summary_words / original_words
            self.store.upsert_summary(
                memory_id=chunk[0].memory_id,
                summary=summary,
                compression_ratio=ratio,
            )
            if create_semantic:
                metadata = {"source_memory_id": chunk[0].memory_id, "kind": "consolidated_summary"}
                self.add_semantic_fact(
                    fact=summary,
                    importance=0.7,
                    semantic_tags=["summary", "consolidated"],
                    metadata=metadata,
                )
            summary_count += 1

        return summary_count

