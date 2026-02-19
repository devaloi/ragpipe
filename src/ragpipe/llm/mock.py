"""Mock LLM provider for testing without API keys."""

from __future__ import annotations

from ragpipe.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic LLM provider that echoes context for testing.

    Returns a formatted response incorporating the provided context,
    enabling end-to-end pipeline tests without external API calls.
    """

    def __init__(self, response_template: str | None = None) -> None:
        self._template = response_template

    def generate(self, prompt: str, context: str = "") -> str:
        if self._template:
            return self._template.format(prompt=prompt, context=context)
        if context:
            return f"Based on the provided context: {context[:200]}\n\nAnswer: {prompt}"
        return f"Answer: {prompt}"
