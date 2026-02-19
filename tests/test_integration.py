"""Integration tests for the full RAG pipeline."""

from pathlib import Path

from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.pipeline import Pipeline
from ragpipe.store.chroma import ChromaStore


def _make_pipeline(name: str) -> tuple[Pipeline, ChromaStore]:
    store = ChromaStore(collection_name=name)
    return Pipeline(
        store=store,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
        chunk_size=200,
        chunk_overlap=20,
    ), store


class TestEndToEnd:
    """Full pipeline integration tests using mock providers."""

    def test_ingest_and_query_text(self):
        pipe, store = _make_pipeline("integ_text")
        pipe.ingest_text(
            "Retrieval-Augmented Generation (RAG) combines retrieval from a "
            "knowledge base with language model generation to produce grounded, "
            "factual answers."
        )
        resp = pipe.query("What is RAG?")
        assert resp.answer
        assert resp.query == "What is RAG?"
        assert len(resp.sources) >= 1
        store.clear()

    def test_ingest_and_query_files(self, tmp_path: Path):
        pipe, store = _make_pipeline("integ_files")
        (tmp_path / "ml.txt").write_text(
            "Machine learning is a subset of artificial intelligence that "
            "enables systems to learn from data without explicit programming.",
            encoding="utf-8",
        )
        (tmp_path / "dl.txt").write_text(
            "Deep learning uses neural networks with many layers to learn "
            "representations of data with multiple levels of abstraction.",
            encoding="utf-8",
        )
        count = pipe.ingest_directory(tmp_path, "*.txt")
        assert count == 2

        resp = pipe.query("What is machine learning?", top_k=2)
        assert resp.answer
        assert len(resp.sources) == 2
        store.clear()

    def test_multiple_ingestions(self):
        pipe, store = _make_pipeline("integ_multi")
        pipe.ingest_text("First batch of documents about Python.")
        pipe.ingest_text("Second batch of documents about JavaScript.")
        assert pipe.document_count == 2

        resp = pipe.query("Python")
        assert resp.answer
        store.clear()

    def test_large_document_chunking_and_retrieval(self):
        pipe, store = _make_pipeline("integ_large")
        text = (
            "Paragraph about embeddings. " * 20
            + "Paragraph about vector databases. " * 20
            + "Paragraph about language models. " * 20
        )
        count = pipe.ingest_text(text)
        assert count > 1

        resp = pipe.query("embeddings", top_k=3)
        assert resp.answer
        assert len(resp.sources) == 3
        store.clear()

    def test_query_with_metadata(self):
        pipe, store = _make_pipeline("integ_meta")
        pipe.ingest_text("Content about databases.", metadata={"source": "db_guide.txt"})
        resp = pipe.query("databases", top_k=1)
        assert resp.sources[0].chunk.metadata.get("source") == "db_guide.txt"
        store.clear()

    def test_response_includes_context_in_answer(self):
        pipe, store = _make_pipeline("integ_ctx")
        pipe.ingest_text("ChromaDB is an open-source embedding database for AI applications.")
        resp = pipe.query("What is ChromaDB?")
        assert "ChromaDB" in resp.answer
        store.clear()

    def test_pipeline_with_custom_llm_template(self):
        store = ChromaStore(collection_name="integ_template")
        pipe = Pipeline(
            store=store,
            embedding_provider=MockEmbeddingProvider(dimension=32),
            llm_provider=MockLLMProvider(response_template="Answer: {context}"),
            chunk_size=200,
            chunk_overlap=20,
        )
        pipe.ingest_text("The sky is blue.")
        resp = pipe.query("What color is the sky?")
        assert "sky is blue" in resp.answer
        store.clear()
