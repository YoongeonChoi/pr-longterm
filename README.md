# Cognitive Memory Engine for LLM Agents

MemGPT-inspired hierarchical memory system for long-context reasoning, multi-session conversations, and autonomous memory management.

## Implemented End-to-End Scope

- Working / Episodic / Semantic / Archival-oriented memory flow
- Persistent memory store (SQLite) with required schema tables
- Hybrid retrieval: vector + keyword + recency + importance + graph boost
- Query rewriting (`RAG`, `LLM` short-form expansion)
- Context budget manager with recursive long-context compression
- Document ingestion and chunked semantic memory indexing
- Cognitive agent with multi-session memory continuity
- FastAPI endpoints for memory, ingest, chat, consolidation, benchmark
- Evaluation harness for:
  - long document QA
  - long conversation memory
  - knowledge recall

## Architecture

```text
User
  -> Agent Controller / Cognitive Agent
  -> Context Manager
  -> Memory Manager
  -> Hybrid Retriever
  -> LLM
```

Memory layers:

1. Working memory (session runtime state)
2. Episodic memory (turn-by-turn interaction traces)
3. Semantic memory (fact/document chunks and extracted knowledge)
4. Archival memory (summaries/consolidated records)

## Repository Structure

```text
agent/
api/
compression/
docs/
evaluation/
experiments/
memory/
retrieval/
tests/
```

## Setup

```bash
python -m pip install -e .[dev]
```

## Test & Coverage

```bash
python -m pytest -q
```

Current status:

- Tests: 20 passed
- Coverage: 93.03% (`>=80%` target satisfied)

## Run

API server:

```bash
uvicorn api.app:app --reload
```

Demo:

```bash
python -m experiments.run_demo
```

## API Endpoints

- `GET /health`
- `POST /memory/write`
- `GET /memory/search`
- `POST /document/ingest`
- `POST /chat`
- `POST /memory/consolidate`
- `GET /session/history`
- `GET /benchmark/quick`

## Key Design Docs

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Implementation Status](docs/implementation-status.md)
- [ADR-0001](docs/adr/0001-memory-hierarchy-and-local-store.md)

