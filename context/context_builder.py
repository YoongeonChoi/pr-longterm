from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from compression.strategies import ContextCompressor, count_tokens
from context.context_ranker import ContextRanker, RankedContext
from context.token_budget import TokenBudget
from retrieval.types import RetrievedMemory


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, limit: int = 5) -> list[RetrievedMemory]:
        ...


@dataclass(slots=True)
class BuiltContext:
    prompt: str
    total_tokens: int
    selected: list[RankedContext] = field(default_factory=list)
    was_compressed: bool = False


class ContextBuilder:
    """Prompt assembly with ranking and token budgeting."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        ranker: ContextRanker | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.retriever = retriever
        self.ranker = ranker or ContextRanker()
        self.compressor = compressor or ContextCompressor()

    def build(
        self,
        query: str,
        conversation: list[str] | None = None,
        budget: TokenBudget | None = None,
        retrieval_limit: int = 10,
    ) -> BuiltContext:
        conversation = conversation or []
        budget = budget or TokenBudget(max_tokens=2048)
        candidates = self.retriever.retrieve(query=query, limit=retrieval_limit)
        ranked = self.ranker.rank(query, candidates)

        blocks = []
        selected: list[RankedContext] = []
        if conversation:
            blocks.append("Conversation:\n" + "\n".join(conversation[-20:]))

        current_text = "\n\n".join(blocks)
        for item in ranked:
            block = f"[{item.score:.3f}] ({item.memory.memory_type}) {item.memory.content}"
            tentative = "\n\n".join([current_text, "Memories:\n" + block]) if current_text else "Memories:\n" + block
            if count_tokens(tentative) > budget.context_budget:
                break
            current_text = tentative
            selected.append(item)

        prompt = f"User Query:\n{query}\n\n{current_text}".strip()
        tokens = count_tokens(prompt)
        if tokens <= budget.context_budget:
            return BuiltContext(prompt=prompt, total_tokens=tokens, selected=selected, was_compressed=False)

        compressed = self.compressor.compress([current_text], token_budget=budget.context_budget - count_tokens(query))
        prompt = f"User Query:\n{query}\n\nCompressed Context:\n{compressed.compressed_text}"
        prompt = budget.truncate(prompt)
        return BuiltContext(
            prompt=prompt,
            total_tokens=count_tokens(prompt),
            selected=selected,
            was_compressed=True,
        )

