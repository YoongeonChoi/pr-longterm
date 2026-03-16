from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.ingestion import DocumentIngestor, IngestionResult
from memory.models import MemoryRecord, MemoryType
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel


@dataclass(slots=True)
class SemanticMemory:
    """Stores long-lived facts and document knowledge."""

    store: MemoryStore
    embedding_model: SimpleEmbeddingModel

    @classmethod
    def create(cls, store: MemoryStore, dimensions: int = 64) -> "SemanticMemory":
        return cls(store=store, embedding_model=SimpleEmbeddingModel(dimensions=dimensions))

    def add_fact(
        self,
        fact: str,
        importance: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.store.add_memory(
            memory_type=MemoryType.SEMANTIC,
            content=fact,
            importance_score=importance,
            semantic_tags=tags or ["fact", "knowledge"],
            embedding=self.embedding_model.embed_text(fact),
            metadata=metadata or {},
        )

    def ingest_document(
        self,
        title: str,
        text: str,
        chunk_size_words: int = 240,
        overlap_words: int = 40,
        base_metadata: dict[str, str] | None = None,
    ) -> IngestionResult:
        ingestor = DocumentIngestor(store=self.store, embedding_model=self.embedding_model)
        return ingestor.ingest_document(
            title=title,
            text=text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            base_metadata=base_metadata,
        )

    def search_facts(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        candidates = self.store.query_keyword(query, limit=limit, memory_types=[MemoryType.SEMANTIC])
        records: list[MemoryRecord] = []
        for item in candidates:
            record = self.store.get_memory(item.memory_id)
            if record is not None:
                records.append(record)
        return records

