# ADR 0001: Hierarchical Memory with Local SQLite Store

- Date: 2026-03-16
- Status: Accepted

## Context

The project needs a robust baseline that can be tested locally without requiring external services. We also need a schema that maps directly to episodic/semantic workflows and can later migrate to dedicated stores.

## Decision

1. Use a four-layer memory hierarchy:
   - Working memory (in-memory runtime state)
   - Episodic memory (event log)
   - Semantic memory (distilled facts)
   - Archival snapshots (history + summaries)
2. Use SQLite as the first persistence backend.
3. Store embeddings in `memory_embeddings` and metadata in dedicated tables to preserve extensibility.
4. Implement hybrid retrieval score:

   `score = alpha * relevance + beta * recency + gamma * importance`

## Consequences

### Positive

- Fast local setup and deterministic testing
- Clear migration path to pgvector/Qdrant/Weaviate
- Schema supports future graph consolidation and summarization

### Negative

- SQLite is not ideal for very high concurrency
- Vector operations are limited vs specialized vector databases

## Follow-up

- Add backend abstraction for vector stores in Month 5
- Add migration scripts for PostgreSQL/pgvector compatibility

