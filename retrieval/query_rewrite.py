from __future__ import annotations

import re


DEFAULT_EXPANSIONS = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "ctx": "context",
    "mem": "memory",
}


class QueryRewriter:
    def __init__(self, expansions: dict[str, str] | None = None) -> None:
        self.expansions = {**DEFAULT_EXPANSIONS, **(expansions or {})}

    def rewrite(self, query: str) -> str:
        lowered = query.lower()
        additions: list[str] = []
        for short, expanded in self.expansions.items():
            if re.search(rf"\b{re.escape(short)}\b", lowered):
                additions.append(expanded)
        if not additions:
            return query
        suffix = " ".join(sorted(set(additions)))
        return f"{query} {suffix}".strip()

