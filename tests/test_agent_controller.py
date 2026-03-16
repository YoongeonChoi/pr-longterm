from __future__ import annotations

from agent.context_manager import ContextManager
from agent.controller import AgentController
from compression.strategies import ContextCompressor
from memory.manager import MemoryManager
from memory.store import MemoryStore


class StubRetriever:
    def retrieve(self, query: str, limit: int = 5):
        return []


class StubLLM:
    def generate(self, prompt: str) -> str:
        return f"LLM_RESPONSE::{prompt.splitlines()[0]}"


def test_agent_controller_executes_loop_and_persists_episode(tmp_path) -> None:
    store = MemoryStore(str(tmp_path / "memory.db"))
    memory_manager = MemoryManager(store)
    context_manager = ContextManager(
        token_budget=200,
        retriever=StubRetriever(),
        compressor=ContextCompressor(),
    )
    controller = AgentController(
        llm=StubLLM(),
        context_manager=context_manager,
        memory_manager=memory_manager,
    )

    result = controller.run("What did we discuss?")

    assert result.response.startswith("LLM_RESPONSE::")
    episodes = store.list_memory(limit=10)
    assert episodes
    assert "What did we discuss?" in episodes[0].content

