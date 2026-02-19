"""Tests for LLM providers."""

import pytest

from ragpipe.llm.base import LLMProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.llm.openai import OpenAILLMProvider


class TestMockLLMProvider:
    def test_implements_interface(self):
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_generate_without_context(self):
        provider = MockLLMProvider()
        result = provider.generate("What is RAG?")
        assert "What is RAG?" in result

    def test_generate_with_context(self):
        provider = MockLLMProvider()
        result = provider.generate("What is RAG?", context="RAG stands for Retrieval-Augmented")
        assert "RAG stands for Retrieval-Augmented" in result
        assert "What is RAG?" in result

    def test_custom_template(self):
        provider = MockLLMProvider(response_template="Q: {prompt} | C: {context}")
        result = provider.generate("test", context="ctx")
        assert result == "Q: test | C: ctx"

    def test_template_without_context(self):
        provider = MockLLMProvider(response_template="Q: {prompt} | C: {context}")
        result = provider.generate("test")
        assert result == "Q: test | C: "


class TestOpenAILLMProvider:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            OpenAILLMProvider()

    def test_accepts_explicit_key(self):
        provider = OpenAILLMProvider(api_key="sk-test-key")
        assert isinstance(provider, LLMProvider)
