from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent.cognitive_agent import CognitiveMemoryAgent
from context.context_builder import ContextBuilder
from context.context_ranker import ContextRanker
from context.token_budget import TokenBudget
from evaluation.long_conversation_test import run_long_conversation_test
from evaluation.long_document_qa import run_long_document_qa
from memory.archival_memory import ArchivalMemory
from memory.episodic_memory import EpisodicMemory
from memory.manager import MemoryManager
from memory.memory_consolidation import MemoryConsolidation
from memory.models import MemoryType
from memory.semantic_memory import SemanticMemory
from memory.store import MemoryStore
from memory.working_memory import WorkingMemory
from retrieval.embedding import SimpleEmbeddingModel
from retrieval.hybrid import HybridRetriever


def test_memory_layer_modules_and_consolidation(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    manager = MemoryManager(store)
    working = WorkingMemory()
    episodic = EpisodicMemory(store)
    semantic = SemanticMemory.create(store, dimensions=16)
    archival = ArchivalMemory(store)
    consolidation = MemoryConsolidation(store=store, manager=manager)

    working.set("s1", "goal", "memory hierarchy")
    assert working.get("s1", "goal") == "memory hierarchy"

    event_id = episodic.write_event("s1", "hello", "we discuss memory")
    events = episodic.read_session_events("s1", limit=10)
    assert any(item.memory_id == event_id for item in events)

    fact_id = semantic.add_fact("Working memory stores current task state.")
    facts = semantic.search_facts("current task")
    assert any(item.memory_id == fact_id for item in facts)

    archive_id = archival.archive_snapshot("summary", source_memory_id=fact_id, compression_ratio=0.2)
    assert archive_id
    assert archival.list_archives(limit=5)

    store.add_memory(
        memory_type=MemoryType.EPISODIC,
        content="old low-importance note",
        importance_score=0.05,
        timestamp=datetime.now(timezone.utc) - timedelta(days=30),
    )
    pruned = consolidation.prune_low_importance_memory(min_importance=0.1, older_than_days=14)
    assert pruned >= 1

    embed = [0.5, 0.5, 0.5]
    store.add_memory(MemoryType.SEMANTIC, "duplicate fact one", embedding=embed, importance_score=0.4)
    store.add_memory(MemoryType.SEMANTIC, "duplicate fact one", embedding=embed, importance_score=0.5)
    merged = consolidation.merge_redundant_memory(similarity_threshold=0.99)
    assert merged >= 1


def test_context_builder_applies_ranking_and_budget(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    embedder = SimpleEmbeddingModel(dimensions=16)
    retriever = HybridRetriever(store=store, embedding_model=embedder)

    store.add_memory(
        MemoryType.SEMANTIC,
        "Memory hierarchy contains working episodic semantic and archival memory.",
        importance_score=0.9,
        embedding=embedder.embed_text("memory hierarchy architecture"),
    )
    store.add_memory(
        MemoryType.SEMANTIC,
        "Shopping memo for apples and milk.",
        importance_score=0.3,
        embedding=embedder.embed_text("shopping memo apples"),
    )

    builder = ContextBuilder(retriever=retriever, ranker=ContextRanker())
    built = builder.build(
        query="Explain memory hierarchy",
        conversation=["user: tell me architecture details"],
        budget=TokenBudget(max_tokens=120, reserved_response_tokens=30, reserved_system_tokens=10),
        retrieval_limit=5,
    )
    assert built.total_tokens <= 120
    assert built.selected
    assert "memory hierarchy" in built.prompt.lower()


def test_evaluation_entrypoints(tmp_path) -> None:
    agent = CognitiveMemoryAgent(db_path=str(tmp_path / "memory.db"))
    conv = run_long_conversation_test(agent=agent, session_id="bench-1", turns=20)
    assert conv.turns == 20

    doc = run_long_document_qa(
        agent=agent,
        session_id="bench-2",
        document_text=" ".join(["context compression memory"] * 4000),
        question="What are the key topics?",
        expected_keywords=["context", "memory"],
    )
    assert 0.0 <= doc.accuracy <= 1.0

