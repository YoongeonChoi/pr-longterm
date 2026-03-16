from __future__ import annotations

from agent.cognitive_agent import CognitiveMemoryAgent
from agent.llm import HeuristicLLM
from evaluation.benchmarks import evaluate_long_conversation, evaluate_long_document_qa


def run_demo() -> None:
    agent = CognitiveMemoryAgent(db_path=":memory:", llm=HeuristicLLM())

    ingest = agent.ingest_document(
        session_id="demo-session",
        title="Memory Architecture",
        text=(
            "Hierarchical memory includes working memory for current reasoning, "
            "episodic memory for interaction history, semantic memory for facts, "
            "and archival storage for compressed long-term records. "
        )
        * 30,
    )
    print(f"Ingested document chunks: {ingest.total_chunks}")

    first = agent.chat("demo-session", "What does the architecture include?")
    print("Q1:", first.response)

    second = agent.chat("demo-session", "What did we discuss just now?")
    print("Q2:", second.response)

    convo = evaluate_long_conversation(agent=agent, session_id="demo-bench", turns=30)
    print("Conversation benchmark:", convo)

    long_doc_eval = evaluate_long_document_qa(
        agent=agent,
        session_id="demo-doc",
        document_text=" ".join(["context compression memory"] * 12000),
        question="What are the core topics?",
        expected_keywords=["context", "compression", "memory"],
    )
    print("Long document benchmark:", long_doc_eval)


if __name__ == "__main__":
    run_demo()

