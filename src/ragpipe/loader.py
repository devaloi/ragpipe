"""Document loaders for reading files into Document objects."""

from __future__ import annotations

from pathlib import Path

from ragpipe.types import Document


def load_text(path: str | Path) -> Document:
    """Load a plain text file as a Document."""
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return Document(content=content, metadata={"source": str(p), "format": "text"})


def load_markdown(path: str | Path) -> Document:
    """Load a Markdown file as a Document."""
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return Document(content=content, metadata={"source": str(p), "format": "markdown"})


def load_directory(path: str | Path, glob_pattern: str = "*.txt") -> list[Document]:
    """Load all matching files from a directory."""
    p = Path(path)
    docs: list[Document] = []
    for file_path in sorted(p.glob(glob_pattern)):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext == ".md":
                docs.append(load_markdown(file_path))
            else:
                docs.append(load_text(file_path))
    return docs
