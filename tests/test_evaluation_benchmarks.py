from __future__ import annotations

from agent.cognitive_agent import CognitiveMemoryAgent
from agent.llm import HeuristicLLM
from evaluation.benchmarks import (
    evaluate_knowledge_recall,
    evaluate_long_conversation,
    evaluate_long_document_qa,
)
from retrieval.query_rewrite import QueryRewriter


def test_evaluation_scenarios_run_and_return_metrics(tmp_path) -> None:
    agent = CognitiveMemoryAgent(db_path=str(tmp_path / "memory.db"), llm=HeuristicLLM(), query_rewriter=QueryRewriter())

    recall = evaluate_knowledge_recall(
        memory_manager=agent.memory_manager,
        retriever=agent.retriever,
        facts=["Paris is the capital of France.", "Memory scoring uses relevance recency importance."],
    )
    assert 0.0 <= recall.accuracy <= 1.0
    assert recall.token_usage > 0

    conversation = evaluate_long_conversation(agent=agent, session_id="bench-session", turns=50)
    assert 0.0 <= conversation.memory_recall_accuracy <= 1.0
    assert conversation.turns == 50

    doc_eval = evaluate_long_document_qa(
        agent=agent,
        session_id="doc-session",
        document_text=" ".join(["context compression"] * 5000),
        question="What is the dominant topic?",
        expected_keywords=["context", "compression"],
    )
    assert 0.0 <= doc_eval.accuracy <= 1.0
    assert doc_eval.token_usage > 0

