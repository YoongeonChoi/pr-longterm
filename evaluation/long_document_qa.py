from __future__ import annotations

from agent.cognitive_agent import CognitiveMemoryAgent
from evaluation.benchmarks import BenchmarkResult, evaluate_long_document_qa


def run_long_document_qa(
    agent: CognitiveMemoryAgent,
    session_id: str,
    document_text: str,
    question: str,
    expected_keywords: list[str],
) -> BenchmarkResult:
    return evaluate_long_document_qa(
        agent=agent,
        session_id=session_id,
        document_text=document_text,
        question=question,
        expected_keywords=expected_keywords,
    )

