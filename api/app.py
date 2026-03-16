from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from agent.cognitive_agent import CognitiveMemoryAgent
from agent.llm import HeuristicLLM
from evaluation.benchmarks import evaluate_long_conversation, evaluate_long_document_qa
from memory.models import MemoryType, parse_memory_type
from retrieval.embedding import SimpleEmbeddingModel


class WriteMemoryRequest(BaseModel):
    memory_type: str = Field(default=MemoryType.EPISODIC.value)
    content: str
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WriteMemoryResponse(BaseModel):
    memory_id: str


class SearchMemoryResponse(BaseModel):
    results: list[dict[str, Any]]


class IngestDocumentRequest(BaseModel):
    session_id: str
    title: str
    text: str
    chunk_size_words: int = Field(default=240, ge=50, le=2000)
    overlap_words: int = Field(default=40, ge=0, le=500)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ConsolidateRequest(BaseModel):
    max_events: int = Field(default=100, ge=1, le=1000)
    chunk_size: int = Field(default=10, ge=1, le=200)


def create_app(db_path: str = "memory_engine.db") -> FastAPI:
    agent = CognitiveMemoryAgent(db_path=db_path, llm=HeuristicLLM())
    store = agent.store
    manager = agent.memory_manager
    embedder = SimpleEmbeddingModel()
    retriever = agent.retriever

    app = FastAPI(title="Cognitive Memory Engine API", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/memory/write", response_model=WriteMemoryResponse)
    def write_memory(payload: WriteMemoryRequest) -> WriteMemoryResponse:
        mem_type = parse_memory_type(payload.memory_type)
        memory_id = store.add_memory(
            memory_type=mem_type,
            content=payload.content,
            importance_score=payload.importance_score,
            semantic_tags=payload.semantic_tags,
            embedding=embedder.embed_text(payload.content),
            metadata=payload.metadata,
        )
        return WriteMemoryResponse(memory_id=memory_id)

    @app.get("/memory/search", response_model=SearchMemoryResponse)
    def search_memory(
        query: str = Query(..., min_length=1),
        limit: int = Query(5, ge=1, le=50),
    ) -> SearchMemoryResponse:
        hits = retriever.retrieve(query=query, limit=limit)
        if not hits:
            fallback = manager.keyword_search(query=query, limit=limit)
            return SearchMemoryResponse(
                results=[
                    {
                        "memory_id": item.memory_id,
                        "content": item.content,
                        "memory_type": item.memory_type.value,
                        "score": item.keyword_score,
                    }
                    for item in fallback
                ]
            )
        return SearchMemoryResponse(
            results=[
                {
                    **asdict(item),
                    "timestamp": item.timestamp.isoformat(),
                }
                for item in hits
            ]
        )

    @app.post("/document/ingest")
    def ingest_document(payload: IngestDocumentRequest) -> dict[str, Any]:
        result = agent.ingest_document(
            session_id=payload.session_id,
            title=payload.title,
            text=payload.text,
            chunk_size_words=payload.chunk_size_words,
            overlap_words=payload.overlap_words,
        )
        return {
            "document_id": result.document_id,
            "total_chunks": result.total_chunks,
            "total_tokens": result.total_tokens,
        }

    @app.post("/chat")
    def chat(payload: ChatRequest) -> dict[str, Any]:
        result = agent.chat(payload.session_id, payload.message)
        return {
            "session_id": result.session_id,
            "response": result.response,
            "context_tokens": result.context_tokens,
            "was_compressed": result.was_compressed,
            "observations": result.observations,
        }

    @app.post("/memory/consolidate")
    def consolidate(payload: ConsolidateRequest) -> dict[str, int]:
        count = manager.consolidate(max_events=payload.max_events, chunk_size=payload.chunk_size, create_semantic=True)
        return {"summaries_created": count}

    @app.get("/session/history")
    def session_history(session_id: str = Query(...), limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        records = agent.get_session_history(session_id=session_id, limit=limit)
        return {
            "session_id": session_id,
            "count": len(records),
            "records": [
                {
                    "memory_id": item.memory_id,
                    "memory_type": item.memory_type.value,
                    "content": item.content,
                    "timestamp": item.timestamp.isoformat(),
                }
                for item in records
            ],
        }

    @app.get("/benchmark/quick")
    def benchmark_quick() -> dict[str, Any]:
        conversation = evaluate_long_conversation(agent=agent, session_id="bench-api", turns=25)
        long_doc = evaluate_long_document_qa(
            agent=agent,
            session_id="bench-doc",
            document_text=" ".join(["context compression memory architecture"] * 8000),
            question="What are the main topics?",
            expected_keywords=["context", "compression", "memory"],
        )
        return {
            "conversation": asdict(conversation),
            "long_document": asdict(long_doc),
        }

    return app


app = create_app()

