"""Core data types for the ragpipe RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Document:
    """A source document to be processed by the RAG pipeline."""

    content: str
    metadata: dict[str, str] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Chunk:
    """A chunk of text split from a document, ready for embedding."""

    content: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Query:
    """A user query for the RAG pipeline."""

    text: str
    top_k: int = 5


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with a relevance score."""

    chunk: Chunk
    score: float


@dataclass
class Response:
    """The final response from the RAG pipeline."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
