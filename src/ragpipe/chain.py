"""Retrieval chain — query the vector store and generate a response."""

from __future__ import annotations

from ragpipe.embeddings.base import EmbeddingProvider
from ragpipe.llm.base import LLMProvider
from ragpipe.store.base import VectorStore
from ragpipe.types import Query, Response, RetrievedChunk


class RetrievalChain:
    """Connects a vector store, embedding provider, and LLM to answer queries.

    Flow: embed query → search vector store → build context → generate answer.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> None:
        self._store = store
        self._embeddings = embedding_provider
        self._llm = llm_provider

    def retrieve(self, query: Query) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self._embeddings.embed(query.text)
        return self._store.query(query_embedding, top_k=query.top_k)

    def run(self, query: Query) -> Response:
        """Retrieve context and generate an answer."""
        retrieved = self.retrieve(query)
        context = self._build_context(retrieved)
        answer = self._llm.generate(query.text, context=context)
        return Response(answer=answer, sources=retrieved, query=query.text)

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not chunks:
            return ""
        parts: list[str] = []
        for i, rc in enumerate(chunks, 1):
            parts.append(f"[{i}] {rc.chunk.content}")
        return "\n\n".join(parts)
