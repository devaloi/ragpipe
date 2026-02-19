# Build ragpipe — Retrieval-Augmented Generation Pipeline

You are building a **portfolio project** for a Senior AI Engineer's public GitHub. This is a centerpiece AI project — it must be exceptionally impressive, clean, and production-grade. Read these docs before writing any code:

1. **`P05-python-rag-pipeline.md`** — Complete project spec: architecture, ingestion pipeline, retrieval strategies (vector, BM25, hybrid, MMR), LLM provider interface, evaluation metrics, CLI reference, phased build plan, commit plan. This is your primary blueprint. Follow it phase by phase.
2. **`github-portfolio.md`** — Portfolio goals and Definition of Done (Level 1 + Level 2). Understand the quality bar.
3. **`github-portfolio-checklist.md`** — Pre-publish checklist. Every item must pass before you're done.

---

## Instructions

### Read first, build second
Read all three docs completely before writing a single line of code. Understand the full RAG pipeline flow: document loading → chunking → embedding → vector storage → retrieval (vector, BM25, hybrid) → MMR re-ranking → prompt construction with context → LLM generation → citation parsing. Understand how the YAML config drives every component, how the evaluation metrics work, and why this project avoids LangChain.

### Follow the phases in order
The project spec has 5 phases. Do them in order:
1. **Ingestion Pipeline** — project setup, document loaders (PDF, markdown, text, HTML), chunking strategies (fixed, recursive, semantic), sentence-transformers embedder with caching, ChromaDB vector store
2. **Retrieval** — vector similarity retriever, BM25 keyword retriever, hybrid retriever with Reciprocal Rank Fusion, MMR re-ranker for diversity
3. **Generation** — LLM provider interface, Ollama/OpenAI/Anthropic adapters, prompt templates with citation instructions, answer generator with streaming
4. **Evaluation** — eval dataset loader, faithfulness/relevance/similarity metrics, eval runner with reporting
5. **CLI + Server + Polish** — CLI commands (ingest, query, serve, eval, status, config), FastAPI HTTP server, pipeline configuration, integration tests, README

### Commit frequently
Follow the commit plan in the spec. Use **conventional commits** (`feat:`, `test:`, `refactor:`, `docs:`, `chore:`). Each commit should be a logical unit.

### Quality non-negotiables
- **No LangChain.** This entire pipeline is built from primitives. Every component — loaders, chunkers, embedders, retrievers, generators — is hand-written. This is what separates a Senior AI Engineer from someone who calls `langchain.load()`.
- **Local-first embeddings.** sentence-transformers with `all-MiniLM-L6-v2` runs locally with zero API keys. The project must be runnable out of the box with just Ollama for LLM.
- **Hybrid search with RRF.** The hybrid retriever runs vector and BM25 in parallel and fuses with Reciprocal Rank Fusion. This is not optional — it's a core differentiator.
- **MMR diversity.** After retrieval, MMR re-ranking prevents redundant chunks. Configurable lambda parameter.
- **Source citations.** Every generated answer must include inline citations mapping back to source documents and page numbers. The prompt template instructs the LLM to cite, and the citation parser extracts references.
- **Streaming responses.** Both CLI and HTTP API support token-by-token streaming from the LLM.
- **Evaluation metrics.** Faithfulness, relevance, answer similarity, context precision, retrieval recall — all implemented and runnable via `ragpipe eval`.
- **Provider-agnostic LLM.** Clean `LLMProvider` protocol with adapters for OpenAI, Anthropic, and Ollama. No SDK dependencies for LLM APIs — use httpx directly.
- **Tests at every layer.** Loaders, chunkers, embedder, store, retrievers, generator, metrics, pipeline integration. Use respx to mock HTTP calls, fixtures for sample documents.
- **Lint clean.** `ruff check` and `ruff format --check` must pass with zero issues.

### What NOT to do
- Don't use LangChain, LlamaIndex, or any RAG framework. Build everything from scratch.
- Don't use OpenAI/Anthropic SDKs. Use httpx for all LLM API calls — this shows you understand the APIs.
- Don't skip hybrid search. Vector-only retrieval is table stakes. Hybrid with RRF is what makes this impressive.
- Don't skip evaluation. A RAG pipeline without metrics is a demo, not a portfolio piece.
- Don't make real API calls in tests. Use respx for HTTP mocking and fixture embeddings for vector operations.
- Don't leave `# TODO` or `# FIXME` comments anywhere.

---

## GitHub Username

The GitHub username is **devaloi**. The repository is `github.com/devaloi/ragpipe`. Python package name is `ragpipe`.

## Start

Read the three docs. Then begin Phase 1 from `P05-python-rag-pipeline.md`.
