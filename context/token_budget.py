from __future__ import annotations

from dataclasses import dataclass

from compression.strategies import count_tokens


@dataclass(slots=True)
class TokenBudget:
    max_tokens: int
    reserved_response_tokens: int = 512
    reserved_system_tokens: int = 64

    @property
    def context_budget(self) -> int:
        return max(32, self.max_tokens - self.reserved_response_tokens - self.reserved_system_tokens)

    def fits(self, text: str) -> bool:
        return count_tokens(text) <= self.context_budget

    def truncate(self, text: str) -> str:
        budget = self.context_budget
        words = text.split()
        if len(words) <= budget:
            return text
        return " ".join(words[:budget])

