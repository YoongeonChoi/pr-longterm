from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agent.context_manager import ContextManager
from memory.manager import MemoryManager


class LLMProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass(slots=True)
class AgentRunResult:
    plan: list[str]
    response: str
    context_tokens: int
    was_compressed: bool
    observations: list[str] = field(default_factory=list)


class AgentController:
    def __init__(self, llm: LLMProtocol, context_manager: ContextManager, memory_manager: MemoryManager) -> None:
        self.llm = llm
        self.context_manager = context_manager
        self.memory_manager = memory_manager

    def _plan(self, user_input: str) -> list[str]:
        return [
            "Clarify user intent and target outcome.",
            "Retrieve relevant episodic and semantic memories.",
            "Synthesize final answer with concise rationale.",
        ]

    def run(self, user_input: str, conversation_history: list[str] | None = None) -> AgentRunResult:
        plan = self._plan(user_input)
        context = self.context_manager.build_context(
            query=user_input,
            conversation=conversation_history or [],
            retrieval_limit=5,
        )
        prompt = (
            f"Task: {user_input}\n"
            f"Plan:\n- " + "\n- ".join(plan) + "\n\n"
            f"Context:\n{context.prompt}"
        )
        response = self.llm.generate(prompt)
        self.memory_manager.write_interaction(
            user_input=user_input,
            agent_output=response,
            importance=0.75,
            semantic_tags=["interaction", "agent_loop"],
            metadata={"loop": "plan-execute-observe-revise"},
        )
        observations = [
            f"context_tokens={context.total_tokens}",
            f"compressed={context.was_compressed}",
        ]
        return AgentRunResult(
            plan=plan,
            response=response,
            context_tokens=context.total_tokens,
            was_compressed=context.was_compressed,
            observations=observations,
        )

