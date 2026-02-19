"""OpenAI embedding provider (requires OPENAI_API_KEY)."""

from __future__ import annotations

import os

import httpx

from ragpipe.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMENSION = 1536


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings via the OpenAI API."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        dimension: int = _DEFAULT_DIMENSION,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._dimension = dimension
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required")

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"input": texts, "model": self._model},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    @property
    def dimension(self) -> int:
        return self._dimension
