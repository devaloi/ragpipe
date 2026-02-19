"""Tests for the retrieval chain."""

from ragpipe.chain import RetrievalChain
from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.store.chroma import ChromaStore
from ragpipe.types import Chunk, Query


def _build_chain(collection: str) -> tuple[RetrievalChain, ChromaStore, MockEmbeddingProvider]:
    emb = MockEmbeddingProvider(dimension=32)
    store = ChromaStore(collection_name=collection)
    llm = MockLLMProvider()
    chain = RetrievalChain(store=store, embedding_provider=emb, llm_provider=llm)
    return chain, store, emb


def _seed_store(store: ChromaStore, emb: MockEmbeddingProvider, texts: list[str]) -> list[Chunk]:
    chunks = [
        Chunk(content=t, doc_id="doc1", chunk_index=i) for i, t in enumerate(texts)
    ]
    vectors = emb.embed_batch(texts)
    store.add(chunks, vectors)
    return chunks


class TestRetrievalChain:
    def test_retrieve_returns_chunks(self):
        chain, store, emb = _build_chain("test_chain_retrieve")
        _seed_store(store, emb, ["Python is great", "RAG pipelines work"])
        results = chain.retrieve(Query(text="Python is great", top_k=1))
        assert len(results) == 1
        assert "Python" in results[0].chunk.content
        store.clear()

    def test_run_returns_response(self):
        chain, store, emb = _build_chain("test_chain_run")
        _seed_store(store, emb, ["Embeddings capture meaning", "Vectors enable search"])
        resp = chain.run(Query(text="What are embeddings?"))
        assert resp.answer
        assert resp.query == "What are embeddings?"
        assert len(resp.sources) > 0
        store.clear()

    def test_run_with_empty_store(self):
        chain, store, _ = _build_chain("test_chain_empty")
        resp = chain.run(Query(text="anything"))
        assert resp.answer
        assert resp.sources == []
        store.clear()

    def test_context_building(self):
        chain, store, emb = _build_chain("test_chain_ctx")
        _seed_store(store, emb, ["fact one", "fact two", "fact three"])
        retrieved = chain.retrieve(Query(text="fact one", top_k=2))
        context = chain._build_context(retrieved)
        assert "[1]" in context
        assert "[2]" in context
        store.clear()

    def test_empty_context(self):
        context = RetrievalChain._build_context([])
        assert context == ""
