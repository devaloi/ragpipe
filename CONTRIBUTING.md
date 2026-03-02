# Contributing to ragpipe

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/devaloi/ragpipe.git
cd ragpipe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
make test         # Run all tests (no real API calls — uses mock embeddings)
make lint         # Run ruff + mypy
make all          # Lint, type-check, test
```

All tests use mock embeddings and an in-process ChromaDB, so no API keys are needed.

## Project Structure

```
src/ragpipe/
  pipeline.py     # Main RAGPipeline class
  embeddings/     # Embedding providers (mock, OpenAI)
  retrieval/      # Vector store and similarity search
  chunking/       # Document splitters
tests/            # Unit and integration tests
```

## Adding a New Embedding Provider

1. Implement `BaseEmbedder` in `src/ragpipe/embeddings/`
2. Register it in `EmbedderFactory`
3. Add tests in `tests/embeddings/`

## Pull Request Guidelines

- One feature or fix per PR
- Run `make all` before submitting
- Add tests for new code
- Update README if adding a new feature

## Reporting Issues

Open a GitHub issue with Python version, steps to reproduce, and expected vs actual behavior.
