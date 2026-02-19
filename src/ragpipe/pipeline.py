"""RAG pipeline — end-to-end document ingestion, retrieval, and generation."""

from __future__ import annotations

from pathlib import Path

from ragpipe.chain import RetrievalChain
from ragpipe.embeddings.base import EmbeddingProvider
from ragpipe.llm.base import LLMProvider
from ragpipe.loader import load_directory, load_text
from ragpipe.splitter import TextSplitter
from ragpipe.store.base import VectorStore
from ragpipe.types import Chunk, Document, Query, Response


class Pipeline:
    """Full RAG pipeline: load → split → embed → store → retrieve → generate.

    Supports both programmatic and file-based document ingestion.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self._store = store
        self._embeddings = embedding_provider
        self._splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._chain = RetrievalChain(
            store=store,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    def ingest_text(self, text: str, metadata: dict[str, str] | None = None) -> int:
        """Ingest a raw text string. Returns the number of chunks stored."""
        doc = Document(content=text, metadata=metadata or {})
        return self._ingest_documents([doc])

    def ingest_file(self, path: str | Path) -> int:
        """Ingest a single file. Returns the number of chunks stored."""
        doc = load_text(str(path))
        return self._ingest_documents([doc])

    def ingest_directory(self, path: str | Path, glob_pattern: str = "*.txt") -> int:
        """Ingest all matching files from a directory. Returns the number of chunks stored."""
        docs = load_directory(str(path), glob_pattern)
        return self._ingest_documents(docs)

    def query(self, text: str, top_k: int = 5) -> Response:
        """Query the pipeline and get a generated response with sources."""
        return self._chain.run(Query(text=text, top_k=top_k))

    @property
    def document_count(self) -> int:
        """Return the total number of chunks in the vector store."""
        return self._store.count()

    def _ingest_documents(self, documents: list[Document]) -> int:
        """Split, embed, and store a list of documents."""
        chunks: list[Chunk] = self._splitter.split_many(documents)
        if not chunks:
            return 0
        embeddings = self._embeddings.embed_batch([c.content for c in chunks])
        self._store.add(chunks, embeddings)
        return len(chunks)
