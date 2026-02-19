"""Tests for document loaders."""

from pathlib import Path

import pytest

from ragpipe.loader import load_directory, load_markdown, load_text


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("Hello, world!", encoding="utf-8")
    return f


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome content.", encoding="utf-8")
    return f


class TestLoadText:
    def test_loads_content(self, text_file: Path):
        doc = load_text(text_file)
        assert doc.content == "Hello, world!"

    def test_metadata_source(self, text_file: Path):
        doc = load_text(text_file)
        assert doc.metadata["source"] == str(text_file)
        assert doc.metadata["format"] == "text"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/file.txt")


class TestLoadMarkdown:
    def test_loads_content(self, md_file: Path):
        doc = load_markdown(md_file)
        assert "# Title" in doc.content

    def test_metadata_format(self, md_file: Path):
        doc = load_markdown(md_file)
        assert doc.metadata["format"] == "markdown"


class TestLoadDirectory:
    def test_loads_multiple_files(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")
        docs = load_directory(tmp_path, "*.txt")
        assert len(docs) == 2

    def test_empty_directory(self, tmp_path: Path):
        docs = load_directory(tmp_path, "*.txt")
        assert docs == []

    def test_glob_filter(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.md").write_text("# b", encoding="utf-8")
        docs = load_directory(tmp_path, "*.md")
        assert len(docs) == 1
        assert docs[0].metadata["format"] == "markdown"
