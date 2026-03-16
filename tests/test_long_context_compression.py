from __future__ import annotations

from compression.long_context import LongContextCompressor


def test_long_context_compressor_reduces_large_text_to_target_budget() -> None:
    compressor = LongContextCompressor()
    source_text = " ".join(["memory"] * 100000)

    result = compressor.compress(
        source_text,
        target_tokens=8000,
        chunk_tokens=2000,
    )

    assert result.original_tokens >= 100000
    assert result.compressed_tokens <= 8000
    assert result.was_compressed is True
    assert result.recursion_depth >= 1

