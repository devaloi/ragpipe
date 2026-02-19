"""CLI interface for the ragpipe RAG pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from ragpipe.embeddings.mock import MockEmbeddingProvider
from ragpipe.llm.mock import MockLLMProvider
from ragpipe.pipeline import Pipeline
from ragpipe.store.chroma import ChromaStore


def _build_pipeline(
    chroma_path: str | None,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Pipeline:
    """Build a pipeline with the configured providers."""
    store = ChromaStore(
        collection_name=collection,
        persist_directory=chroma_path,
    )
    embedding_provider = _get_embedding_provider()
    llm_provider = _get_llm_provider()
    return Pipeline(
        store=store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _get_embedding_provider():  # type: ignore[no-untyped-def]
    """Return the appropriate embedding provider."""
    try:
        from ragpipe.embeddings.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider()
    except (ValueError, ImportError):
        click.echo("⚠ No OPENAI_API_KEY set, using mock embedding provider", err=True)
        return MockEmbeddingProvider()


def _get_llm_provider():  # type: ignore[no-untyped-def]
    """Return the appropriate LLM provider."""
    try:
        from ragpipe.llm.openai import OpenAILLMProvider

        return OpenAILLMProvider()
    except (ValueError, ImportError):
        click.echo("⚠ No OPENAI_API_KEY set, using mock LLM provider", err=True)
        return MockLLMProvider()


@click.group()
@click.version_option(package_name="ragpipe")
def main() -> None:
    """ragpipe — A RAG pipeline with embeddings, ChromaDB, and retrieval chain."""


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--glob", "glob_pattern", default="*.txt", help="File glob pattern.")
@click.option("--chroma-path", default="./chroma_data", help="ChromaDB storage path.")
@click.option("--collection", default="documents", help="Collection name.")
@click.option("--chunk-size", default=500, type=int, help="Chunk size in characters.")
@click.option("--chunk-overlap", default=50, type=int, help="Chunk overlap in characters.")
def ingest(
    path: str,
    glob_pattern: str,
    chroma_path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Ingest documents from a file or directory into the vector store."""
    pipe = _build_pipeline(chroma_path, collection, chunk_size, chunk_overlap)
    p = Path(path)
    if p.is_dir():
        count = pipe.ingest_directory(p, glob_pattern)
        click.echo(f"✓ Ingested {count} chunks from directory: {p}")
    else:
        count = pipe.ingest_file(p)
        click.echo(f"✓ Ingested {count} chunks from file: {p}")


@main.command()
@click.argument("question")
@click.option("--top-k", default=5, type=int, help="Number of chunks to retrieve.")
@click.option("--chroma-path", default="./chroma_data", help="ChromaDB storage path.")
@click.option("--collection", default="documents", help="Collection name.")
def query(question: str, top_k: int, chroma_path: str, collection: str) -> None:
    """Query the RAG pipeline with a question."""
    pipe = _build_pipeline(chroma_path, collection, chunk_size=500, chunk_overlap=50)
    resp = pipe.query(question, top_k=top_k)
    click.echo(f"\n📝 Answer:\n{resp.answer}\n")
    if resp.sources:
        click.echo("📚 Sources:")
        for i, src in enumerate(resp.sources, 1):
            score = f"{src.score:.3f}"
            preview = src.chunk.content[:80].replace("\n", " ")
            click.echo(f"  [{i}] (score: {score}) {preview}...")


@main.command()
@click.option("--chroma-path", default="./chroma_data", help="ChromaDB storage path.")
@click.option("--collection", default="documents", help="Collection name.")
def status(chroma_path: str, collection: str) -> None:
    """Show the number of chunks in the vector store."""
    store = ChromaStore(collection_name=collection, persist_directory=chroma_path)
    click.echo(f"📊 Collection '{collection}': {store.count()} chunks")
