from __future__ import annotations

from dataclasses import dataclass
import time

from agent.cognitive_agent import CognitiveMemoryAgent
from memory.manager import MemoryManager
from memory.models import MemoryType
from retrieval.hybrid import HybridRetriever


@dataclass(slots=True)
class BenchmarkResult:
    task_name: str
    accuracy: float
    token_usage: int
    latency_ms: float


@dataclass(slots=True)
class ConversationBenchmarkResult:
    task_name: str
    turns: int
    memory_recall_accuracy: float
    coherence: float
    latency_ms: float


def evaluate_knowledge_recall(
    memory_manager: MemoryManager,
    retriever: HybridRetriever,
    facts: list[str],
    query_prefix: str = "Recall:",
) -> BenchmarkResult:
    start = time.perf_counter()
    for fact in facts:
        memory_manager.store.add_memory(
            memory_type=MemoryType.SEMANTIC,
            content=fact,
            importance_score=0.8,
            semantic_tags=["benchmark", "fact"],
        )

    correct = 0
    token_usage = 0
    for fact in facts:
        query = f"{query_prefix} {fact.split()[0]}"
        hits = retriever.retrieve(query, limit=3)
        token_usage += len(query.split()) + sum(len(hit.content.split()) for hit in hits)
        if any(fact.lower() in hit.content.lower() for hit in hits):
            correct += 1

    latency_ms = (time.perf_counter() - start) * 1000
    accuracy = correct / len(facts) if facts else 0.0
    return BenchmarkResult(
        task_name="knowledge_recall",
        accuracy=accuracy,
        token_usage=token_usage,
        latency_ms=latency_ms,
    )


def evaluate_long_conversation(
    agent: CognitiveMemoryAgent,
    session_id: str,
    turns: int = 500,
) -> ConversationBenchmarkResult:
    start = time.perf_counter()
    turns = max(10, turns)
    anchor_fact = "The user's project codename is Atlas."
    agent.chat(session_id, f"Remember this fact: {anchor_fact}")
    mention_hits = 0
    coherence_hits = 0

    for idx in range(turns - 2):
        question = f"Turn {idx}: continue with memory-planning details."
        result = agent.chat(session_id, question)
        response = result.response.lower()
        if "memory" in response:
            coherence_hits += 1
        if "atlas" in response:
            mention_hits += 1

    final = agent.chat(session_id, "What is the user's project codename?")
    if "atlas" in final.response.lower():
        mention_hits += 1

    latency_ms = (time.perf_counter() - start) * 1000
    return ConversationBenchmarkResult(
        task_name="long_conversation",
        turns=turns,
        memory_recall_accuracy=mention_hits / turns,
        coherence=coherence_hits / turns,
        latency_ms=latency_ms,
    )


def evaluate_long_document_qa(
    agent: CognitiveMemoryAgent,
    session_id: str,
    document_text: str,
    question: str,
    expected_keywords: list[str],
) -> BenchmarkResult:
    start = time.perf_counter()
    result = agent.analyze_long_document(
        session_id=session_id,
        title="LongDocumentBenchmark",
        document_text=document_text,
        question=question,
    )
    response_lower = result.response.lower()
    expected = [keyword.lower() for keyword in expected_keywords]
    accuracy = 0.0
    if expected:
        matched = sum(1 for keyword in expected if keyword in response_lower)
        accuracy = matched / len(expected)

    token_usage = result.context_tokens + len(question.split())
    latency_ms = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        task_name="long_document_qa",
        accuracy=accuracy,
        token_usage=token_usage,
        latency_ms=latency_ms,
    )

