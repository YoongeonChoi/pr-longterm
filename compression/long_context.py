from __future__ import annotations

from dataclasses import dataclass

from compression.strategies import count_tokens, hierarchical_summarize


def _word_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    words = [word for word in text.split() if word.strip()]
    chunk_tokens = max(50, chunk_tokens)
    overlap_tokens = max(0, min(overlap_tokens, chunk_tokens // 3))
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_tokens)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(0, end - overlap_tokens)
    return chunks


@dataclass(slots=True)
class LongContextCompressionResult:
    original_tokens: int
    compressed_tokens: int
    compressed_text: str
    recursion_depth: int
    was_compressed: bool


class LongContextCompressor:
    def compress(
        self,
        text: str,
        target_tokens: int = 8000,
        chunk_tokens: int = 2000,
        overlap_tokens: int = 200,
        max_depth: int = 8,
    ) -> LongContextCompressionResult:
        target_tokens = max(50, target_tokens)
        current_text = text
        original_tokens = count_tokens(text)
        current_tokens = original_tokens
        depth = 0

        while current_tokens > target_tokens and depth < max_depth:
            chunks = _word_chunks(current_text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            # Summarize each chunk into roughly one-quarter size to force convergence.
            summaries = [hierarchical_summarize(chunk, target_tokens=max(30, chunk_tokens // 4)) for chunk in chunks]
            current_text = "\n".join(summaries)
            current_tokens = count_tokens(current_text)
            depth += 1
            # Tighten chunk size each round for more aggressive compression when needed.
            chunk_tokens = max(200, int(chunk_tokens * 0.7))
            overlap_tokens = max(20, int(overlap_tokens * 0.5))

        if current_tokens > target_tokens:
            current_text = " ".join(current_text.split()[:target_tokens])
            current_tokens = count_tokens(current_text)

        return LongContextCompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            compressed_text=current_text,
            recursion_depth=depth,
            was_compressed=original_tokens > target_tokens,
        )

