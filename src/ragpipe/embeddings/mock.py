"""Mock embedding provider for testing without API keys."""

from __future__ import annotations

import hashlib
import math

from ragpipe.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider that generates vectors from text hashes.

    Produces consistent embeddings for the same input text, enabling
    reproducible tests without external API calls.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding from a hash of the text."""
        digest = hashlib.sha256(text.encode()).hexdigest()
        raw = [int(digest[i : i + 2], 16) / 255.0 for i in range(0, len(digest), 2)]
        # Extend or truncate to match dimension
        while len(raw) < self._dimension:
            raw.extend(raw)
        raw = raw[: self._dimension]
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]
        return raw

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension
