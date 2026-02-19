"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for language model generation."""

    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> str:
        """Generate a response given a prompt and optional context.

        Args:
            prompt: The user question or instruction.
            context: Retrieved context to ground the response.

        Returns:
            The generated text response.
        """
