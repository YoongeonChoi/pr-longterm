from __future__ import annotations

from memory.ingestion import DocumentIngestor
from memory.store import MemoryStore
from retrieval.embedding import SimpleEmbeddingModel


def test_document_ingestion_splits_and_persists_chunks(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    ingestor = DocumentIngestor(store=store, embedding_model=SimpleEmbeddingModel(dimensions=32))

    text = " ".join(["long-context-memory"] * 1200)
    result = ingestor.ingest_document(
        title="Long Context Design",
        text=text,
        chunk_size_words=200,
        overlap_words=20,
    )

    assert result.total_chunks > 1
    stored = store.search_by_metadata("document_id", result.document_id, limit=200)
    assert len(stored) == result.total_chunks
    assert all("Long Context Design" == item.metadata["title"] for item in stored)

