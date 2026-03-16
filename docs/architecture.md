# Cognitive Memory Engine Architecture

## 1. Objectives

This system extends LLM agents with hierarchical memory to support:

- Long document reasoning (100k+ token inputs)
- Multi-session conversational continuity
- High-precision recall of prior facts/events
- Autonomous memory management (store/retrieve/consolidate/compress)

## 2. High-Level Flow

```text
User -> AgentController -> ContextManager -> MemoryManager -> Retriever -> LLM
```

## 3. Memory Hierarchy

### Working Memory

- Scope: current request/task loop
- Lifetime: short
- Data: task state, active goals, intermediate reasoning artifacts

### Episodic Memory

- Scope: interaction-level events
- Lifetime: medium to long
- Data: user turns, tool outputs, task outcomes, action traces

### Semantic Memory

- Scope: distilled facts/knowledge
- Lifetime: long
- Data: extracted entities, stable facts, document summaries

### Archival Storage

- Scope: raw historical records and compacted snapshots
- Lifetime: long-term
- Data: historical events and summary checkpoints

## 4. Core Subsystems

### Agent Controller

- Loop: `plan -> execute -> observe -> revise`
- Handles tool orchestration and response synthesis

### Context Manager

- Token budgeting
- Context assembly from memory + current input
- Compression fallback for oversized contexts

### Memory Manager

- Unified API for writing/reading working, episodic, semantic memory
- Consolidation and pruning hooks

### Retriever

- Hybrid retrieval:
  - Vector similarity
  - Keyword overlap
  - Recency weighting
  - Importance weighting
  - Query rewriting for acronym/short-form expansion
  - Graph boost from memory relation edges
- Final score:

```text
score = alpha * relevance + beta * recency + gamma * importance
```

### Compression

- Hierarchical summarization
- Chunk-level semantic compression
- Topic extraction for navigation
- Recursive long-context compression loop (for very large documents)

### Cognitive Agent

- Session-aware conversation handling
- Document ingestion + semantic indexing
- Periodic consolidation after multiple turns
- Benchmark-ready orchestration surface

## 5. Persistence Schema

Implemented in SQLite for local development and testing.

- `memory_events`
- `memory_embeddings`
- `memory_summaries`
- `memory_graph`
- `memory_metadata`

## 6. Development Phases (6 Months)

- Month 1: Baseline RAG + ingestion + vector retrieval (implemented)
- Month 2: Persistent episodic/semantic memory APIs (implemented)
- Month 3: Hierarchical memory with prioritization/summarization (implemented)
- Month 4: Context compression for long-document reasoning (implemented)
- Month 5: Advanced retrieval (hybrid ranking + graph memory) (implemented)
- Month 6: Full cognitive agent (multi-session + long-context assistant) (implemented)

## 7. Quality Gates

- Unit + integration tests
- Type hints and modular boundaries
- Coverage target: 80%+
- Measurable benchmark harness for recall, token use, latency
