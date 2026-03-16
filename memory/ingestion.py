from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from memory.models import MemoryType
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel


def _chunk_words(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunk_size = max(20, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    total_chunks: int
    total_tokens: int


class DocumentIngestor:
    def __init__(self, store: MemoryStore, embedding_model: SimpleEmbeddingModel | None = None) -> None:
        self.store = store
        self.embedding_model = embedding_model or SimpleEmbeddingModel()

    def ingest_document(
        self,
        title: str,
        text: str,
        chunk_size_words: int = 240,
        overlap_words: int = 40,
        importance_score: float = 0.7,
        base_metadata: dict[str, str] | None = None,
    ) -> IngestionResult:
        words = [word for word in text.split() if word.strip()]
        document_id = str(uuid4())
        chunks = _chunk_words(words, chunk_size=chunk_size_words, overlap=overlap_words)
        metadata = base_metadata or {}

        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "document_id": document_id,
                "title": title,
                "chunk_index": str(idx),
                "chunk_total": str(len(chunks)),
            }
            self.store.add_memory(
                memory_type=MemoryType.SEMANTIC,
                content=chunk,
                importance_score=importance_score,
                semantic_tags=["document", "chunk", title.lower().replace(" ", "_")],
                embedding=self.embedding_model.embed_text(chunk),
                metadata=chunk_metadata,
            )

        return IngestionResult(document_id=document_id, total_chunks=len(chunks), total_tokens=len(words))

