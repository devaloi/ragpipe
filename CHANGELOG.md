# Changelog

All notable changes to ragpipe are documented here.

## [0.2.0] - 2026-02-20

### Added
- Batch 9 depth improvements: retrieval scoring visibility and query tracing
- GitHub Actions CI with Python matrix (3.11, 3.12)
- Mock embedding provider for fast, deterministic tests

### Changed
- Improved chunking strategy: configurable overlap and min-chunk-size
- Retrieval chain now returns source document metadata alongside results

## [0.1.0] - 2026-02-18

### Added
- `RAGPipeline` class with load, embed, store, query workflow
- Document loaders: text, markdown, directory
- Configurable chunk splitter (size + overlap)
- Embedding providers: mock (default), OpenAI
- ChromaDB vector store with cosine similarity search
- Retrieval chain: embed query → top-k search → context builder
- CLI: `ragpipe ingest`, `ragpipe query`
- MIT License
