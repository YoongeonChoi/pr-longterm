from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = np.array(list(vec_a), dtype=float)
    b = np.array(list(vec_b), dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, value))


class SimpleEmbeddingModel:
    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=float)
        tokens = _tokenize(text)
        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            magnitude = 1.0 + (len(token) / 10.0)
            vector[index] += sign * magnitude

        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector.tolist()
        return (vector / norm).tolist()

