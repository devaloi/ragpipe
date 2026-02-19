"""Tests for the RAG pipeline."""

from pathlib import Path

from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.pipeline import Pipeline
from ragpipe.store.chroma import ChromaStore


def _make_pipeline(collection: str) -> tuple[Pipeline, ChromaStore]:
    store = ChromaStore(collection_name=collection)
    pipe = Pipeline(
        store=store,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
        chunk_size=100,
        chunk_overlap=10,
    )
    return pipe, store


class TestPipeline:
    def test_ingest_text(self):
        pipe, store = _make_pipeline("test_pipe_ingest")
        count = pipe.ingest_text("Some important document content for testing purposes.")
        assert count == 1
        assert pipe.document_count == 1
        store.clear()

    def test_ingest_empty_text(self):
        pipe, store = _make_pipeline("test_pipe_empty")
        count = pipe.ingest_text("")
        assert count == 0
        store.clear()

    def test_ingest_file(self, tmp_path: Path):
        pipe, store = _make_pipeline("test_pipe_file")
        f = tmp_path / "doc.txt"
        f.write_text("Document loaded from file.", encoding="utf-8")
        count = pipe.ingest_file(f)
        assert count == 1
        store.clear()

    def test_ingest_directory(self, tmp_path: Path):
        pipe, store = _make_pipeline("test_pipe_dir")
        (tmp_path / "a.txt").write_text("First document.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Second document.", encoding="utf-8")
        count = pipe.ingest_directory(tmp_path, "*.txt")
        assert count == 2
        store.clear()

    def test_query_after_ingest(self):
        pipe, store = _make_pipeline("test_pipe_query")
        pipe.ingest_text("Python is a popular programming language used in data science.")
        resp = pipe.query("What is Python?")
        assert resp.answer
        assert resp.query == "What is Python?"
        assert len(resp.sources) > 0
        store.clear()

    def test_query_empty_store(self):
        pipe, store = _make_pipeline("test_pipe_query_empty")
        resp = pipe.query("anything")
        assert resp.answer
        assert resp.sources == []
        store.clear()

    def test_ingest_text_with_metadata(self):
        pipe, store = _make_pipeline("test_pipe_meta")
        count = pipe.ingest_text("content", metadata={"source": "manual"})
        assert count == 1
        store.clear()

    def test_large_document_chunking(self):
        pipe, store = _make_pipeline("test_pipe_large")
        text = "word " * 500  # ~2500 chars, should produce multiple chunks
        count = pipe.ingest_text(text)
        assert count > 1
        store.clear()
