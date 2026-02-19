"""Tests for vector stores."""

from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.store.base import VectorStore
from ragpipe.store.chroma import ChromaStore
from ragpipe.types import Chunk


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(content=f"chunk {i} content", doc_id="doc1", chunk_index=i)
        for i in range(n)
    ]


class TestChromaStore:
    def test_implements_interface(self):
        store = ChromaStore(collection_name="test_iface")
        assert isinstance(store, VectorStore)
        store.clear()

    def test_add_and_count(self):
        store = ChromaStore(collection_name="test_add")
        emb = MockEmbeddingProvider(dimension=32)
        chunks = _make_chunks(3)
        vectors = emb.embed_batch([c.content for c in chunks])
        store.add(chunks, vectors)
        assert store.count() == 3
        store.clear()

    def test_add_empty(self):
        store = ChromaStore(collection_name="test_empty")
        store.add([], [])
        assert store.count() == 0
        store.clear()

    def test_query_returns_results(self):
        store = ChromaStore(collection_name="test_query")
        emb = MockEmbeddingProvider(dimension=32)
        chunks = _make_chunks(5)
        vectors = emb.embed_batch([c.content for c in chunks])
        store.add(chunks, vectors)

        query_vec = emb.embed("chunk 0 content")
        results = store.query(query_vec, top_k=3)
        assert len(results) == 3
        assert results[0].chunk.content == "chunk 0 content"
        assert results[0].score >= results[1].score
        store.clear()

    def test_query_top_k_limits(self):
        store = ChromaStore(collection_name="test_topk")
        emb = MockEmbeddingProvider(dimension=32)
        chunks = _make_chunks(10)
        vectors = emb.embed_batch([c.content for c in chunks])
        store.add(chunks, vectors)

        results = store.query(emb.embed("chunk 0 content"), top_k=2)
        assert len(results) == 2
        store.clear()

    def test_clear(self):
        store = ChromaStore(collection_name="test_clear")
        emb = MockEmbeddingProvider(dimension=32)
        chunks = _make_chunks(3)
        vectors = emb.embed_batch([c.content for c in chunks])
        store.add(chunks, vectors)
        assert store.count() == 3
        store.clear()
        assert store.count() == 0

    def test_metadata_preserved(self):
        store = ChromaStore(collection_name="test_meta")
        emb = MockEmbeddingProvider(dimension=32)
        chunk = Chunk(
            content="with metadata",
            doc_id="d1",
            chunk_index=0,
            metadata={"source": "file.txt"},
        )
        vec = emb.embed_batch([chunk.content])
        store.add([chunk], vec)

        results = store.query(emb.embed("with metadata"), top_k=1)
        assert results[0].chunk.doc_id == "d1"
        assert results[0].chunk.metadata.get("source") == "file.txt"
        store.clear()
