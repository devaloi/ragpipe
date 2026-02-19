"""ChromaDB vector store implementation."""

from __future__ import annotations

import chromadb

from ragpipe.store.base import VectorStore
from ragpipe.types import Chunk, RetrievedChunk


class ChromaStore(VectorStore):
    """Vector store backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
    ) -> None:
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "chunk_index": str(c.chunk_index),
                    **c.metadata,
                }
                for c in chunks
            ],
        )

    def query(
        self, embedding: list[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()) or 1,
            include=["documents", "metadatas", "distances"],
        )
        retrieved: list[RetrievedChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc_id = meta.pop("doc_id", "")
            chunk_index = int(meta.pop("chunk_index", "0"))
            content = results["documents"][0][i] if results["documents"] else ""
            distance = results["distances"][0][i] if results["distances"] else 0.0
            score = 1.0 - distance  # cosine distance → similarity

            chunk = Chunk(
                content=content,
                doc_id=doc_id,
                chunk_index=chunk_index,
                metadata=meta,
                chunk_id=chunk_id,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
