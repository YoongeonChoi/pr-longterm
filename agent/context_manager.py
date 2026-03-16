from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from compression.strategies import ContextCompressor, count_tokens
from retrieval.types import RetrievedMemory


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, limit: int = 5) -> list[RetrievedMemory]:
        ...


@dataclass(slots=True)
class ContextPackage:
    prompt: str
    total_tokens: int
    was_compressed: bool
    memories: list[RetrievedMemory] = field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        token_budget: int,
        retriever: RetrieverProtocol,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self.token_budget = token_budget
        self.retriever = retriever
        self.compressor = compressor or ContextCompressor()

    def build_context(
        self,
        query: str,
        conversation: list[str] | None = None,
        retrieval_limit: int = 5,
    ) -> ContextPackage:
        conversation = conversation or []
        memories = self.retriever.retrieve(query, limit=retrieval_limit)

        conversation_text = "\n".join(conversation)
        memory_text = "\n".join(
            f"[{idx + 1}] ({memory.memory_type}) {memory.content}"
            for idx, memory in enumerate(memories)
        )
        raw_prompt = (
            "You are a memory-augmented agent.\n"
            f"User Query:\n{query}\n\n"
            f"Conversation History:\n{conversation_text}\n\n"
            f"Retrieved Memory:\n{memory_text}\n"
        )
        raw_tokens = count_tokens(raw_prompt)
        if raw_tokens <= self.token_budget:
            return ContextPackage(
                prompt=raw_prompt,
                total_tokens=raw_tokens,
                was_compressed=False,
                memories=memories,
            )

        reserved_query_tokens = max(10, count_tokens(query) + 10)
        compressed_budget = max(10, self.token_budget - reserved_query_tokens)
        compressed = self.compressor.compress(
            text_blocks=[conversation_text, memory_text],
            token_budget=compressed_budget,
        )
        compressed_prompt = (
            "You are a memory-augmented agent.\n"
            f"User Query:\n{query}\n\n"
            f"Compressed Context:\n{compressed.compressed_text}\n"
        )
        final_tokens = count_tokens(compressed_prompt)
        if final_tokens > self.token_budget:
            compressed_prompt = " ".join(compressed_prompt.split()[: self.token_budget])
            final_tokens = count_tokens(compressed_prompt)
        return ContextPackage(
            prompt=compressed_prompt,
            total_tokens=final_tokens,
            was_compressed=True,
            memories=memories,
        )

