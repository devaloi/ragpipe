"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragpipe.types import Chunk, RetrievedChunk


class VectorStore(ABC):
    """Interface for vector storage and similarity search."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings to the store."""

    @abstractmethod
    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Find the most similar chunks to the given embedding."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of items in the store."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the store."""
