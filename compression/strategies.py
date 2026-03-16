from __future__ import annotations

from dataclasses import dataclass
import re


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "about",
    "into",
    "have",
    "will",
    "your",
    "user",
    "agent",
}


def count_tokens(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def extract_topics(text: str, top_n: int = 6) -> list[str]:
    terms = re.findall(r"[a-zA-Z]{4,}", text.lower())
    frequencies: dict[str, int] = {}
    for term in terms:
        if term in STOPWORDS:
            continue
        frequencies[term] = frequencies.get(term, 0) + 1
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:top_n]]


def hierarchical_summarize(text: str, target_tokens: int) -> str:
    target_tokens = max(10, target_tokens)
    words = text.split()
    if len(words) <= target_tokens:
        return text

    topic_prefix = ""
    topics = extract_topics(text)
    if topics:
        topic_prefix = "TOPICS: " + ", ".join(topics) + "\n"

    head_size = max(1, target_tokens // 3)
    tail_size = max(1, target_tokens // 3)
    body_size = max(1, target_tokens - head_size - tail_size - 1)

    head = words[:head_size]
    body = words[len(words) // 2 : len(words) // 2 + body_size]
    tail = words[-tail_size:]
    summary = topic_prefix + " ".join([*head, "...", *body, "...", *tail])
    summary_words = summary.split()
    if len(summary_words) > target_tokens:
        summary = " ".join(summary_words[:target_tokens])
    return summary


@dataclass(slots=True)
class CompressionResult:
    total_tokens: int
    compressed_text: str
    was_compressed: bool


class ContextCompressor:
    def compress(self, text_blocks: list[str], token_budget: int) -> CompressionResult:
        budget = max(10, token_budget)
        merged = "\n\n".join(block for block in text_blocks if block.strip())
        merged_tokens = count_tokens(merged)
        if merged_tokens <= budget:
            return CompressionResult(
                total_tokens=merged_tokens,
                compressed_text=merged,
                was_compressed=False,
            )

        compressed = hierarchical_summarize(merged, target_tokens=budget)
        compressed_tokens = count_tokens(compressed)
        if compressed_tokens > budget:
            compressed = " ".join(compressed.split()[:budget])
            compressed_tokens = count_tokens(compressed)
        return CompressionResult(
            total_tokens=compressed_tokens,
            compressed_text=compressed,
            was_compressed=True,
        )

