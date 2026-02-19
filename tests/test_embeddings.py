"""Tests for embedding providers."""

import math

import pytest

from ragpipe.embeddings.base import EmbeddingProvider
from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.embeddings.openai import OpenAIEmbeddingProvider


class TestMockEmbeddingProvider:
    def test_implements_interface(self):
        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_embed_returns_correct_dimension(self):
        provider = MockEmbeddingProvider(dimension=64)
        vec = provider.embed("hello")
        assert len(vec) == 64

    def test_embed_deterministic(self):
        provider = MockEmbeddingProvider()
        v1 = provider.embed("same text")
        v2 = provider.embed("same text")
        assert v1 == v2

    def test_different_texts_different_embeddings(self):
        provider = MockEmbeddingProvider()
        v1 = provider.embed("text a")
        v2 = provider.embed("text b")
        assert v1 != v2

    def test_embed_normalized(self):
        provider = MockEmbeddingProvider()
        vec = provider.embed("normalize me")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_embed_batch(self):
        provider = MockEmbeddingProvider(dimension=32)
        vectors = provider.embed_batch(["a", "b", "c"])
        assert len(vectors) == 3
        assert all(len(v) == 32 for v in vectors)

    def test_dimension_property(self):
        provider = MockEmbeddingProvider(dimension=256)
        assert provider.dimension == 256


class TestOpenAIEmbeddingProvider:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            OpenAIEmbeddingProvider()

    def test_accepts_explicit_key(self):
        provider = OpenAIEmbeddingProvider(api_key="sk-test-key")
        assert provider.dimension == 1536
