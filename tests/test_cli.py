"""Tests for the CLI interface."""

from pathlib import Path

from click.testing import CliRunner

from ragpipe.cli import main


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ragpipe" in result.output

    def test_ingest_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "Ingest" in result.output

    def test_query_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["query", "--help"])
        assert result.exit_code == 0
        assert "Query" in result.output

    def test_status_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "chunks" in result.output

    def test_ingest_file(self, tmp_path: Path):
        runner = CliRunner()
        f = tmp_path / "test.txt"
        f.write_text("Some document content for testing.", encoding="utf-8")
        chroma_dir = str(tmp_path / "chroma")
        result = runner.invoke(
            main, ["ingest", str(f), "--chroma-path", chroma_dir, "--collection", "cli_test"]
        )
        assert result.exit_code == 0
        assert "Ingested" in result.output

    def test_ingest_directory(self, tmp_path: Path):
        runner = CliRunner()
        (tmp_path / "a.txt").write_text("First file.", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Second file.", encoding="utf-8")
        chroma_dir = str(tmp_path / "chroma")
        result = runner.invoke(
            main,
            ["ingest", str(tmp_path), "--chroma-path", chroma_dir, "--collection", "cli_dir"],
        )
        assert result.exit_code == 0
        assert "Ingested" in result.output

    def test_query_empty(self, tmp_path: Path):
        runner = CliRunner()
        chroma_dir = str(tmp_path / "chroma")
        result = runner.invoke(
            main,
            ["query", "What is Python?", "--chroma-path", chroma_dir, "--collection", "cli_q"],
        )
        assert result.exit_code == 0
        assert "Answer" in result.output

    def test_status(self, tmp_path: Path):
        runner = CliRunner()
        chroma_dir = str(tmp_path / "chroma")
        result = runner.invoke(
            main, ["status", "--chroma-path", chroma_dir, "--collection", "cli_status"]
        )
        assert result.exit_code == 0
        assert "0 chunks" in result.output
