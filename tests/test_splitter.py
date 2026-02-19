"""Tests for the text splitter."""

import pytest

from ragpipe.splitter import TextSplitter
from ragpipe.types import Document


@pytest.fixture
def splitter() -> TextSplitter:
    return TextSplitter(chunk_size=100, chunk_overlap=20)


class TestTextSplitter:
    def test_short_text_single_chunk(self, splitter: TextSplitter):
        doc = Document(content="Short text.")
        chunks = splitter.split(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].chunk_index == 0

    def test_exact_chunk_size(self):
        s = TextSplitter(chunk_size=10, chunk_overlap=0)
        doc = Document(content="0123456789")
        chunks = s.split(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "0123456789"

    def test_overlap(self):
        s = TextSplitter(chunk_size=10, chunk_overlap=3)
        doc = Document(content="abcdefghijklmnopqrst")
        chunks = s.split(doc)
        assert len(chunks) == 3
        assert chunks[0].content == "abcdefghij"
        assert chunks[1].content == "hijklmnopq"
        assert chunks[2].content == "opqrst"

    def test_empty_document(self, splitter: TextSplitter):
        doc = Document(content="")
        chunks = splitter.split(doc)
        assert chunks == []

    def test_whitespace_only(self, splitter: TextSplitter):
        doc = Document(content="   \n  ")
        chunks = splitter.split(doc)
        assert chunks == []

    def test_metadata_propagation(self, splitter: TextSplitter):
        doc = Document(content="Some content", metadata={"source": "test.txt"})
        chunks = splitter.split(doc)
        assert chunks[0].metadata["source"] == "test.txt"

    def test_doc_id_propagation(self, splitter: TextSplitter):
        doc = Document(content="content here")
        chunks = splitter.split(doc)
        assert chunks[0].doc_id == doc.doc_id

    def test_split_many(self, splitter: TextSplitter):
        docs = [Document(content="First doc."), Document(content="Second doc.")]
        chunks = splitter.split_many(docs)
        assert len(chunks) == 2
        assert chunks[0].doc_id != chunks[1].doc_id

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitter(chunk_size=0)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextSplitter(chunk_size=100, chunk_overlap=-1)

    def test_overlap_exceeds_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextSplitter(chunk_size=10, chunk_overlap=10)
