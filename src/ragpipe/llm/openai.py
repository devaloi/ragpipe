"""OpenAI LLM provider (requires OPENAI_API_KEY)."""

from __future__ import annotations

import os

import httpx

from ragpipe.llm.base import LLMProvider

_DEFAULT_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context doesn't contain the answer, say so."
)


class OpenAILLMProvider(LLMProvider):
    """Generate responses via the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._system_prompt = system_prompt
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required")

    def generate(self, prompt: str, context: str = "") -> str:
        user_content = prompt
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {prompt}"

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
