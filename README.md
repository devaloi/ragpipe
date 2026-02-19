# ragpipe

A RAG (Retrieval-Augmented Generation) pipeline with embeddings, ChromaDB, and a retrieval chain — built from clean abstractions with swappable providers.

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│              RAG Pipeline                    │
│                                              │
│  1. Load documents (text, markdown, dir)     │
│  2. Split into chunks (configurable size)    │
│  3. Generate embeddings (mock / OpenAI)      │
│  4. Store in ChromaDB vector store           │
│  5. Query: embed → similarity search         │
│  6. Build context from retrieved chunks      │
│  7. Generate answer via LLM (mock / OpenAI)  │
│                                              │
└─────────────────────────────────────────────┘
    ↓
Response (answer + sources with scores)
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Vector Store | ChromaDB |
| Embeddings | Mock (deterministic) / OpenAI |
| LLM | Mock (deterministic) / OpenAI |
| CLI | Click |
| HTTP | httpx |
| Testing | pytest |
| Linting | Ruff |

## Quick Start

```bash
# Clone and set up
git clone https://github.com/devaloi/ragpipe.git
cd ragpipe
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Ingest documents
ragpipe ingest ./docs/ --glob "*.txt"

# Query the pipeline
ragpipe query "What is retrieval-augmented generation?"

# Check store status
ragpipe status
```

## Project Structure

```
ragpipe/
├── src/ragpipe/
│   ├── __init__.py
│   ├── types.py              # Document, Chunk, Query, Response
│   ├── loader.py             # File loaders (text, markdown, directory)
│   ├── splitter.py           # Text chunking with configurable overlap
│   ├── chain.py              # Retrieval chain (embed → search → generate)
│   ├── pipeline.py           # Full RAG pipeline orchestrator
│   ├── cli.py                # CLI (ingest, query, status)
│   ├── embeddings/
│   │   ├── base.py           # EmbeddingProvider interface
│   │   ├── mock.py           # Deterministic mock (hash-based)
│   │   └── openai.py         # OpenAI text-embedding-3-small
│   ├── store/
│   │   ├── base.py           # VectorStore interface
│   │   └── chroma.py         # ChromaDB implementation
│   └── llm/
│       ├── base.py           # LLMProvider interface
│       ├── mock.py           # Deterministic mock (echo-based)
│       └── openai.py         # OpenAI chat completions
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_types.py
│   ├── test_loader.py
│   ├── test_splitter.py
│   ├── test_embeddings.py
│   ├── test_store.py
│   ├── test_llm.py
│   ├── test_chain.py
│   ├── test_pipeline.py
│   ├── test_cli.py
│   └── test_integration.py
├── pyproject.toml
├── Makefile
├── LICENSE
└── .env.example
```

## Provider Interfaces

All providers implement clean abstract interfaces, making it easy to swap implementations:

```python
from ragpipe.embeddings.base import EmbeddingProvider
from ragpipe.llm.base import LLMProvider
from ragpipe.store.base import VectorStore
```

### Using Mock Providers (no API keys needed)

```python
from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.store.chroma import ChromaStore
from ragpipe.pipeline import Pipeline

pipe = Pipeline(
    store=ChromaStore(),
    embedding_provider=MockEmbeddingProvider(),
    llm_provider=MockLLMProvider(),
)
pipe.ingest_text("Your document content here.")
response = pipe.query("What is in the document?")
print(response.answer)
```

### Using OpenAI Providers

```bash
export OPENAI_API_KEY=sk-your-key-here
```

```python
from ragpipe.embeddings.openai import OpenAIEmbeddingProvider
from ragpipe.llm.openai import OpenAILLMProvider

pipe = Pipeline(
    store=ChromaStore(persist_directory="./chroma_data"),
    embedding_provider=OpenAIEmbeddingProvider(),
    llm_provider=OpenAILLMProvider(),
)
```

## CLI Usage

```bash
# Ingest a single file
ragpipe ingest document.txt

# Ingest a directory with glob pattern
ragpipe ingest ./docs/ --glob "*.md" --chunk-size 1000 --chunk-overlap 100

# Query with custom top-k
ragpipe query "Explain RAG pipelines" --top-k 3

# Check vector store status
ragpipe status --collection my_docs
```

## Running Tests

```bash
# All tests (no API keys required)
python -m pytest -v

# With coverage
python -m pytest --cov=ragpipe

# Lint
ruff check src/ tests/
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (optional) | Falls back to mock |
| `RAGPIPE_CHROMA_PATH` | ChromaDB storage path | `./chroma_data` |
| `RAGPIPE_COLLECTION` | Default collection name | `documents` |

## Design Decisions

- **No LangChain** — built from clean abstractions to demonstrate understanding of RAG internals
- **Mock providers** — all tests run without API keys; mock embedding uses SHA-256 hashing for deterministic, normalized vectors
- **ChromaDB** — lightweight, embeddable vector database with cosine similarity
- **Provider pattern** — abstract base classes for embeddings, LLM, and vector store allow swapping implementations without changing pipeline code

## License

MIT
