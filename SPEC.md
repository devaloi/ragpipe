# P05: ragpipe — Retrieval-Augmented Generation Pipeline

**Catalog ID:** P05 | **Size:** M | **Language:** Python 3.14
**Repo name:** `ragpipe`
**One-liner:** A from-scratch RAG pipeline — document ingestion, intelligent chunking, local embeddings, ChromaDB vector storage, hybrid search (vector + BM25), provider-agnostic LLM answer generation with source citations, streaming responses, and built-in evaluation metrics.

---

## Why This Stands Out

- **No LangChain** — built entirely from primitives, demonstrating deep understanding of every RAG component rather than hiding behind an abstraction layer
- **Hybrid search** — combines dense vector similarity (cosine) with sparse BM25 keyword matching using Reciprocal Rank Fusion, dramatically improving retrieval quality over either method alone
- **MMR diversity** — Maximal Marginal Relevance re-ranking prevents redundant chunks from dominating results, ensuring answer breadth
- **Provider-agnostic LLM** — clean `LLMProvider` interface with adapters for OpenAI, Anthropic, and Ollama — swap models without touching pipeline code
- **Source citations** — every generated answer includes inline citations with document name, page number, and chunk ID so users can verify claims
- **Built-in evaluation** — faithfulness, relevance, and answer similarity metrics let you measure pipeline quality without external tools
- **Local-first embeddings** — sentence-transformers runs locally with no API key required, keeping the project runnable out of the box
- **Production CLI** — `ragpipe ingest`, `ragpipe query`, `ragpipe serve` (HTTP API), `ragpipe eval` — complete workflow from ingestion to serving

---

## Architecture

```
ragpipe/
├── src/
│   └── ragpipe/
│       ├── __init__.py
│       ├── cli.py               # CLI entry point: ingest, query, serve, eval
│       ├── config.py            # YAML pipeline configuration loader
│       ├── pipeline.py          # RAG pipeline orchestrator: ingest + query flows
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── loader.py        # Document loader dispatcher
│       │   ├── pdf_loader.py    # PDF loader (PyMuPDF) with page tracking
│       │   ├── markdown_loader.py  # Markdown loader with section extraction
│       │   ├── text_loader.py   # Plain text loader
│       │   ├── html_loader.py   # HTML loader (BeautifulSoup) with boilerplate removal
│       │   └── chunker.py       # Chunking strategies: fixed, recursive, semantic
│       ├── embed/
│       │   ├── __init__.py
│       │   ├── embedder.py      # Embedder interface + sentence-transformers impl
│       │   └── cache.py         # Embedding cache (SQLite-backed, avoid re-embedding)
│       ├── store/
│       │   ├── __init__.py
│       │   ├── vector_store.py  # VectorStore interface
│       │   └── chroma.py        # ChromaDB adapter with metadata filtering
│       ├── retrieve/
│       │   ├── __init__.py
│       │   ├── retriever.py     # Retriever interface
│       │   ├── vector.py        # Dense vector retriever (top-k cosine similarity)
│       │   ├── bm25.py          # Sparse BM25 keyword retriever
│       │   ├── hybrid.py        # Hybrid retriever (RRF fusion of vector + BM25)
│       │   └── mmr.py           # MMR re-ranker for diversity
│       ├── generate/
│       │   ├── __init__.py
│       │   ├── provider.py      # LLMProvider interface
│       │   ├── openai.py        # OpenAI adapter (httpx, no SDK)
│       │   ├── anthropic.py     # Anthropic adapter (httpx, no SDK)
│       │   ├── ollama.py        # Ollama adapter (local HTTP)
│       │   ├── prompt.py        # Prompt templates with citation instructions
│       │   └── generator.py     # Answer generator with streaming + citation parsing
│       ├── evaluate/
│       │   ├── __init__.py
│       │   ├── metrics.py       # Faithfulness, relevance, answer similarity
│       │   ├── dataset.py       # Eval dataset loader (YAML question/answer pairs)
│       │   └── runner.py        # Eval runner with reporting
│       └── server/
│           ├── __init__.py
│           └── app.py           # Lightweight HTTP API (FastAPI) for query endpoint
├── tests/
│   ├── conftest.py              # Fixtures: sample docs, mock embeddings, mock LLM
│   ├── test_loaders.py          # PDF, markdown, text, HTML loaders
│   ├── test_chunker.py          # Fixed, recursive, semantic chunking
│   ├── test_embedder.py         # Embedding generation and caching
│   ├── test_chroma.py           # ChromaDB store and retrieval
│   ├── test_bm25.py             # BM25 scoring and retrieval
│   ├── test_hybrid.py           # Hybrid search with RRF fusion
│   ├── test_mmr.py              # MMR re-ranking
│   ├── test_generator.py        # Answer generation with citations
│   ├── test_metrics.py          # Evaluation metrics
│   └── test_pipeline.py         # End-to-end pipeline integration
├── testdata/
│   ├── sample.pdf               # Sample PDF for loader tests
│   ├── sample.md                # Sample markdown for loader tests
│   ├── sample.html              # Sample HTML for loader tests
│   ├── sample.txt               # Sample text for loader tests
│   └── eval_dataset.yaml        # Evaluation question/answer pairs
├── configs/
│   ├── default.yaml             # Default pipeline configuration
│   └── fast.yaml                # Fast config (smaller model, fewer chunks)
├── pyproject.toml               # Project metadata, dependencies, CLI scripts
├── .env.example                 # OPENAI_API_KEY, ANTHROPIC_API_KEY (optional)
├── .gitignore
├── .python-version              # 3.14
├── ruff.toml                    # Ruff linter config
├── LICENSE
└── README.md
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `ragpipe ingest <path>` | Ingest documents from file or directory (PDF, MD, TXT, HTML) |
| `ragpipe query "<question>"` | Query the pipeline and get an answer with source citations |
| `ragpipe serve --port 8000` | Start HTTP API server for query endpoint |
| `ragpipe eval <dataset.yaml>` | Run evaluation metrics against a Q&A dataset |
| `ragpipe status` | Show ingested document count, chunk count, store info |
| `ragpipe config --show` | Print active pipeline configuration |

### CLI Flags

| Flag | Applies To | Description |
|------|-----------|-------------|
| `--config <path>` | all | Path to YAML config file (default: `configs/default.yaml`) |
| `--collection <name>` | ingest, query | ChromaDB collection name (default: `default`) |
| `--top-k <n>` | query | Number of chunks to retrieve (default: 5) |
| `--strategy <name>` | query | Retrieval strategy: `vector`, `bm25`, `hybrid` (default: `hybrid`) |
| `--provider <name>` | query, serve | LLM provider: `openai`, `anthropic`, `ollama` (default: `ollama`) |
| `--model <name>` | query, serve | Model name (default: provider-specific) |
| `--stream` | query | Stream the response token-by-token |
| `--no-citations` | query | Disable source citations in output |
| `--chunk-strategy` | ingest | Chunking strategy: `fixed`, `recursive`, `semantic` (default: `recursive`) |
| `--chunk-size <n>` | ingest | Chunk size in tokens (default: 512) |
| `--chunk-overlap <n>` | ingest | Chunk overlap in tokens (default: 50) |

---

## Pipeline Configuration (YAML)

```yaml
embedding:
  model: "all-MiniLM-L6-v2"     # sentence-transformers model
  dimension: 384
  batch_size: 64
  cache_enabled: true

chunking:
  strategy: recursive            # fixed | recursive | semantic
  chunk_size: 512                # tokens
  chunk_overlap: 50              # tokens
  semantic_threshold: 0.85       # for semantic chunking

vector_store:
  provider: chromadb
  persist_directory: ".ragpipe/chroma"
  collection: "default"
  distance_metric: cosine

retrieval:
  strategy: hybrid               # vector | bm25 | hybrid
  top_k: 5
  mmr_enabled: true
  mmr_lambda: 0.7               # 1.0 = pure relevance, 0.0 = pure diversity
  rrf_k: 60                     # RRF constant for rank fusion
  bm25_k1: 1.5
  bm25_b: 0.75

generation:
  provider: ollama               # openai | anthropic | ollama
  model: "llama3.2"
  temperature: 0.1
  max_tokens: 1024
  streaming: true
  citation_style: inline         # inline | footnote
```

---

## Retrieval Strategies

### Vector Similarity (Dense)
```
score = cosine_similarity(query_embedding, chunk_embedding)
return top_k chunks sorted by score descending
```

### BM25 (Sparse)
```
score(q, d) = Σ IDF(qi) · (f(qi, d) · (k1 + 1)) / (f(qi, d) + k1 · (1 - b + b · |d| / avgdl))
where k1 = 1.5, b = 0.75
```

### Hybrid (Reciprocal Rank Fusion)
```
For each chunk appearing in vector OR bm25 results:
  rrf_score = 1/(k + rank_vector) + 1/(k + rank_bm25)
  where k = 60, rank = position in result list (∞ if absent)
Return top_k by rrf_score
```

### MMR Re-ranking
```
MMR(d) = λ · sim(d, query) - (1 - λ) · max(sim(d, d_selected))
Iteratively select chunk maximizing MMR score
λ = 0.7 balances relevance vs. diversity
```

---

## LLM Provider Interface

```python
class LLMProvider(Protocol):
    async def generate(self, prompt: str, *, temperature: float = 0.1,
                       max_tokens: int = 1024) -> str: ...

    async def generate_stream(self, prompt: str, *, temperature: float = 0.1,
                              max_tokens: int = 1024) -> AsyncIterator[str]: ...
```

| Provider | Model Examples | Auth |
|----------|---------------|------|
| OpenAI | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` env var |
| Anthropic | claude-sonnet-4-20250514, claude-haiku | `ANTHROPIC_API_KEY` env var |
| Ollama | llama3.2, mistral, phi-3 | None (local HTTP) |

---

## Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Faithfulness | % of answer claims supported by retrieved chunks | 0.0–1.0 |
| Relevance | Semantic similarity between question and retrieved chunks | 0.0–1.0 |
| Answer Similarity | Semantic similarity between generated answer and ground truth | 0.0–1.0 |
| Context Precision | Proportion of retrieved chunks that are relevant | 0.0–1.0 |
| Retrieval Recall | Proportion of relevant chunks that were retrieved | 0.0–1.0 |

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.14 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB 1.x |
| BM25 | rank-bm25 (pure Python) |
| PDF Parsing | PyMuPDF (fitz) |
| HTML Parsing | BeautifulSoup4 |
| HTTP Client | httpx 0.28 (async, for LLM API calls) |
| HTTP Server | FastAPI + uvicorn (serve mode) |
| CLI | click + rich |
| Config | PyYAML |
| Tokenizer | tiktoken (for token-based chunking) |
| Testing | pytest + pytest-asyncio + respx |
| Linting | ruff |

---

## Phased Build Plan

### Phase 1: Ingestion Pipeline

**1.1 — Project setup**
- `pyproject.toml` with metadata, dependencies, CLI script entry points
- Directory structure, `ruff.toml`, `.gitignore`, `.python-version`
- Install: `sentence-transformers`, `chromadb`, `pymupdf`, `beautifulsoup4`, `rank-bm25`, `httpx`, `click`, `rich`, `pyyaml`, `tiktoken`, `pytest`, `pytest-asyncio`, `respx`

**1.2 — Document loaders**
- `Loader` protocol: `load(path: Path) -> list[Document]` where `Document` is a dataclass with `content`, `metadata` (source, page, title)
- `PDFLoader` — extract text per page with PyMuPDF, track page numbers in metadata
- `MarkdownLoader` — read file, split by headings, track section titles
- `TextLoader` — read file as single document
- `HTMLLoader` — parse with BeautifulSoup, strip nav/footer/script tags, extract main content
- Loader dispatcher: detect file type by extension, select appropriate loader
- Directory ingestion: walk directory, load all supported files
- Tests: each loader with sample files in `testdata/`, verify metadata fields

**1.3 — Chunking strategies**
- `Chunker` protocol: `chunk(documents: list[Document]) -> list[Chunk]` where `Chunk` has `content`, `metadata`, `chunk_id`
- `FixedChunker` — split by token count with overlap (using tiktoken)
- `RecursiveChunker` — split by paragraphs, then sentences, then tokens, respecting boundaries
- `SemanticChunker` — compute sentence embeddings, split where cosine similarity drops below threshold
- All chunkers preserve source metadata and add chunk index
- Tests: chunk sizes within bounds, overlap correct, metadata preserved, semantic splits at topic boundaries

**1.4 — Embedding and storage**
- `Embedder` class wrapping sentence-transformers: `embed(texts: list[str]) -> list[list[float]]`
- Batch embedding with configurable batch size
- Embedding cache: SQLite table mapping content hash → embedding vector, skip re-embedding
- `ChromaStore` adapter: `add(chunks)`, `query(embedding, top_k, filters)`, `delete(collection)`, `count()`
- Metadata filtering: filter by source document, page number, date
- Tests: embed and retrieve round-trip, cache hit avoids model call, metadata filtering

### Phase 2: Retrieval

**2.1 — Vector retriever**
- Embed query → search ChromaDB → return top-k chunks with scores
- Tests: returns ranked results, respects top-k, empty collection returns empty

**2.2 — BM25 retriever**
- Build BM25 index from chunk texts at query time (or cache in-memory)
- Tokenize query and chunks, compute BM25 scores, return top-k
- Tests: keyword-heavy queries score high, irrelevant docs score low

**2.3 — Hybrid retriever with RRF**
- Run vector and BM25 retrievers in parallel
- Fuse results with Reciprocal Rank Fusion: `1/(k + rank_vector) + 1/(k + rank_bm25)`
- Handle chunks appearing in only one result set (assign infinite rank for missing)
- Return top-k by fused score
- Tests: fusion combines both signals, outperforms either alone on mixed queries

**2.4 — MMR re-ranker**
- After retrieval, re-rank with MMR to maximize diversity
- Iterative selection: pick chunk maximizing `λ * relevance - (1-λ) * max_similarity_to_selected`
- Configurable lambda (0.7 default)
- Tests: with duplicate chunks, MMR removes redundancy; lambda=1.0 preserves original order

### Phase 3: Generation

**3.1 — LLM provider interface + Ollama adapter**
- `LLMProvider` protocol: `generate(prompt) -> str`, `generate_stream(prompt) -> AsyncIterator[str]`
- `OllamaProvider`: POST to `http://localhost:11434/api/generate` with httpx
- Streaming: parse NDJSON response lines, yield tokens
- Tests with respx mock: successful generation, streaming tokens, connection error

**3.2 — OpenAI + Anthropic adapters**
- `OpenAIProvider`: POST to chat completions API, parse response, stream via SSE
- `AnthropicProvider`: POST to messages API, parse response, stream via SSE
- API key from environment variable
- Tests with respx mock: success, auth error, rate limit

**3.3 — Prompt templates + citation parsing**
- System prompt instructing LLM to cite sources as `[source:chunk_id]`
- Context formatter: number chunks, include metadata (source, page) in prompt
- Citation parser: extract `[source:X]` references from generated text, map to chunk metadata
- Format citations as inline references or footnotes
- Tests: prompt includes all chunks, citations parsed correctly, unknown citations handled

**3.4 — Answer generator**
- Orchestrate: format context → build prompt → call LLM → parse citations → return answer
- Streaming mode: yield tokens as they arrive, parse citations post-stream
- Return `Answer` dataclass: `text`, `citations: list[Citation]`, `chunks_used`, `latency`
- Tests: end-to-end with mock LLM, citations extracted, streaming yields tokens

### Phase 4: Evaluation

**4.1 — Eval dataset loader**
- YAML format: list of `{question, answer, relevant_chunks?}`
- Load and validate dataset
- Tests: valid dataset, missing fields

**4.2 — Metrics implementation**
- Faithfulness: use LLM to verify each answer claim is supported by context
- Relevance: cosine similarity between query embedding and each chunk embedding
- Answer similarity: cosine similarity between generated answer embedding and ground truth
- Context precision: proportion of retrieved chunks marked relevant by LLM
- Retrieval recall: proportion of expected chunks that appear in retrieved set
- Tests: perfect answer → high scores, irrelevant answer → low scores

**4.3 — Eval runner + reporting**
- Run pipeline on each eval question, compute all metrics
- Aggregate: mean, min, max per metric
- Output: rich table to terminal, JSON report to file
- Tests: runner produces expected metric structure

### Phase 5: CLI + Server + Polish

**5.1 — CLI commands**
- `ragpipe ingest <path>` — load, chunk, embed, store; show progress with rich
- `ragpipe query "<question>"` — retrieve, generate, display answer with citations
- `ragpipe serve --port 8000` — start FastAPI server with `/query` POST endpoint
- `ragpipe eval <dataset.yaml>` — run evaluation, display metric table
- `ragpipe status` — show collection stats
- `ragpipe config --show` — print resolved config

**5.2 — HTTP API server**
- FastAPI app with `/query` POST endpoint: `{question, top_k?, strategy?, stream?}`
- Streaming response via `StreamingResponse` for SSE
- Health check at `/health`
- Tests: query endpoint returns answer, streaming works

**5.3 — Pipeline configuration**
- Load from YAML with sensible defaults
- CLI flags override config values
- Validate config on load
- Tests: defaults applied, overrides work, invalid config rejected

**5.4 — Integration tests**
- Full pipeline: ingest sample docs → query → verify answer contains relevant info
- Hybrid vs. vector-only retrieval comparison
- Evaluation pipeline with sample dataset

**5.5 — README and documentation**
- Badges, install, quick start (3-command demo)
- Architecture diagram
- Pipeline configuration reference
- Retrieval strategy explanation (vector, BM25, hybrid, MMR)
- LLM provider setup (Ollama, OpenAI, Anthropic)
- CLI reference table
- Evaluation guide
- Development commands

---

## Commit Plan

1. `chore: scaffold project with pyproject.toml and directory structure`
2. `feat: add document loaders for PDF, markdown, text, and HTML`
3. `feat: add chunking strategies — fixed, recursive, and semantic`
4. `feat: add sentence-transformers embedder with caching`
5. `feat: add ChromaDB vector store adapter with metadata filtering`
6. `feat: add YAML pipeline configuration loader`
7. `feat: add vector similarity retriever`
8. `feat: add BM25 keyword retriever`
9. `feat: add hybrid retriever with reciprocal rank fusion`
10. `feat: add MMR re-ranker for diversity`
11. `feat: add LLM provider interface with Ollama adapter`
12. `feat: add OpenAI and Anthropic LLM adapters`
13. `feat: add prompt templates with citation parsing`
14. `feat: add answer generator with streaming support`
15. `feat: add evaluation metrics and runner`
16. `feat: add CLI with ingest, query, serve, eval commands`
17. `feat: add FastAPI HTTP server for query endpoint`
18. `test: add end-to-end pipeline integration tests`
19. `refactor: clean up interfaces and configuration handling`
20. `docs: add README with architecture, config reference, and CLI usage`
