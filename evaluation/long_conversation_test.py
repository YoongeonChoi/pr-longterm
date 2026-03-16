from __future__ import annotations

from agent.cognitive_agent import CognitiveMemoryAgent
from evaluation.benchmarks import ConversationBenchmarkResult, evaluate_long_conversation


def run_long_conversation_test(
    agent: CognitiveMemoryAgent,
    session_id: str = "long-conversation-bench",
    turns: int = 500,
) -> ConversationBenchmarkResult:
    return evaluate_long_conversation(agent=agent, session_id=session_id, turns=turns)

