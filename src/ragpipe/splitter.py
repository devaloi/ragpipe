"""Text splitting strategies for chunking documents."""

from __future__ import annotations

from ragpipe.types import Chunk, Document


class TextSplitter:
    """Split documents into chunks with configurable size and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        """Split a document into overlapping chunks."""
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        doc_id=document.doc_id,
                        chunk_index=index,
                        metadata={**document.metadata},
                    )
                )
                index += 1

            if end >= len(text):
                break
            start = end - self.chunk_overlap

        return chunks

    def split_many(self, documents: list[Document]) -> list[Chunk]:
        """Split multiple documents into chunks."""
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.split(doc))
        return chunks
