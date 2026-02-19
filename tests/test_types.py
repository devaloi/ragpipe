"""Tests for core data types."""

from ragpipe.types import Chunk, Document, Query, Response, RetrievedChunk


class TestDocument:
    def test_create_with_defaults(self):
        doc = Document(content="hello world")
        assert doc.content == "hello world"
        assert doc.metadata == {}
        assert len(doc.doc_id) == 32

    def test_create_with_metadata(self):
        doc = Document(content="test", metadata={"source": "file.txt"})
        assert doc.metadata["source"] == "file.txt"

    def test_unique_ids(self):
        d1 = Document(content="a")
        d2 = Document(content="b")
        assert d1.doc_id != d2.doc_id


class TestChunk:
    def test_create(self):
        chunk = Chunk(content="piece", doc_id="doc1", chunk_index=0)
        assert chunk.content == "piece"
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_index == 0
        assert len(chunk.chunk_id) == 32

    def test_with_metadata(self):
        chunk = Chunk(content="x", doc_id="d", chunk_index=1, metadata={"page": "3"})
        assert chunk.metadata["page"] == "3"


class TestQuery:
    def test_defaults(self):
        q = Query(text="what is RAG?")
        assert q.text == "what is RAG?"
        assert q.top_k == 5

    def test_custom_top_k(self):
        q = Query(text="search", top_k=10)
        assert q.top_k == 10


class TestRetrievedChunk:
    def test_create(self):
        chunk = Chunk(content="data", doc_id="d1", chunk_index=0)
        rc = RetrievedChunk(chunk=chunk, score=0.95)
        assert rc.score == 0.95
        assert rc.chunk.content == "data"


class TestResponse:
    def test_create(self):
        resp = Response(answer="The answer is 42.", query="what?")
        assert resp.answer == "The answer is 42."
        assert resp.sources == []
        assert resp.query == "what?"
