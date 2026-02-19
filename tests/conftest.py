"""Shared test fixtures."""

import pytest

from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.store.chroma import ChromaStore


@pytest.fixture
def mock_embeddings() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=32)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def chroma_store() -> ChromaStore:
    store = ChromaStore(collection_name="test_fixture")
    yield store
    store.clear()
