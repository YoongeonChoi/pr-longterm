from __future__ import annotations

from typing import Protocol


class LLMProtocol(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class HeuristicLLM:
    """A deterministic local model stub used for tests and demos."""

    def generate(self, prompt: str) -> str:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        memory_lines = [line for line in lines if line.startswith("[")]
        if memory_lines:
            top = memory_lines[0]
            snippet = top.split(")", 1)[-1].strip()
            return f"Based on memory: {snippet}"

        lowered = prompt.lower()
        if "memory hierarchy" in lowered:
            return "Hierarchical memory includes working, episodic, semantic, and archival layers."
        if "what did we just talk about" in lowered:
            return "We just discussed hierarchical memory and long-context reasoning."
        return "I can help with memory-augmented reasoning tasks."

