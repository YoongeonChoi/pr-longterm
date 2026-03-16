from __future__ import annotations

from agent.cognitive_agent import CognitiveMemoryAgent
from agent.llm import HeuristicLLM


def test_cognitive_agent_supports_multisession_memory_and_doc_reasoning(tmp_path) -> None:
    db_path = str(tmp_path / "memory.db")
    agent = CognitiveMemoryAgent(db_path=db_path, llm=HeuristicLLM())

    ingest = agent.ingest_document(
        session_id="session-a",
        title="System Spec",
        text="Hierarchical memory includes working episodic semantic archival layers. " * 40,
    )
    assert ingest.total_chunks > 0

    first = agent.chat("session-a", "What does hierarchical memory include?")
    assert "hierarchical memory" in first.response.lower()

    second = agent.chat("session-a", "What did we just talk about?")
    assert "hierarchical memory" in second.response.lower() or "working" in second.response.lower()

    history = agent.get_session_history("session-a", limit=10)
    assert history
    assert any("What does hierarchical memory include?" in item.content for item in history)

