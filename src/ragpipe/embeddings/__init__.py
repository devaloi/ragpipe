"""Embedding providers for ragpipe."""

from ragpipe.embeddings.base import EmbeddingProvider
from ragpipe.embeddings.mock import MockEmbeddingProvider

__all__ = ["EmbeddingProvider", "MockEmbeddingProvider"]
