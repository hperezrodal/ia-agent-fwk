"""Tests for the file reader built-in tool."""

import pytest

from ia_agent_fwk.tools.base import ToolContext
from ia_agent_fwk.tools.builtin.file_reader import FileReaderInput, FileReaderOutput, FileReaderTool
from ia_agent_fwk.tools.exceptions import ToolExecutionError


@pytest.fixture
def ctx():
    return ToolContext(execution_id="test-file-reader")


class TestFileReaderSuccess:
    async def test_read_existing_file(self, tmp_path, ctx):
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)

        assert isinstance(result, FileReaderOutput)
        assert result.content == "Hello, World!"
        assert result.file_size == 13
        assert result.encoding == "utf-8"

    async def test_read_empty_file(self, tmp_path, ctx):
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)

        assert isinstance(result, FileReaderOutput)
        assert result.content == ""
        assert result.file_size == 0


class TestFileNotFound:
    async def test_file_not_found(self, tmp_path, ctx):
        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolExecutionError, match="not found"):
            await tool.execute(FileReaderInput(file_path=str(tmp_path / "nonexistent.txt")), ctx)


class TestPathSandboxing:
    async def test_path_outside_allowed_directories(self, tmp_path, ctx):
        # Create a file outside the allowed directory
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir()
        inner_dir = tmp_path / "inner"
        inner_dir.mkdir()
        test_file = outer_dir / "secret.txt"
        test_file.write_text("secret data")

        tool = FileReaderTool(allowed_directories=[str(inner_dir)])
        with pytest.raises(ToolExecutionError, match="outside allowed directories"):
            await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)

    async def test_dotdot_traversal(self, tmp_path, ctx):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("secret")

        tool = FileReaderTool(allowed_directories=[str(allowed_dir)])
        with pytest.raises(ToolExecutionError, match="outside allowed directories"):
            await tool.execute(FileReaderInput(file_path=str(allowed_dir / ".." / "secret.txt")), ctx)

    async def test_symlink_escape(self, tmp_path, ctx):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        secret_file = secret_dir / "data.txt"
        secret_file.write_text("secret data")

        # Create symlink inside allowed_dir pointing to secret_dir
        symlink = allowed_dir / "link_to_secret"
        try:
            symlink.symlink_to(secret_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        tool = FileReaderTool(allowed_directories=[str(allowed_dir)])
        with pytest.raises(ToolExecutionError, match="outside allowed directories"):
            await tool.execute(FileReaderInput(file_path=str(symlink)), ctx)


class TestFileSizeLimit:
    async def test_file_exceeds_max_size(self, tmp_path, ctx):
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 1000)

        tool = FileReaderTool(allowed_directories=[str(tmp_path)], max_file_size=500)
        with pytest.raises(ToolExecutionError, match="too large"):
            await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)


class TestEncodingHandling:
    async def test_utf8_file(self, tmp_path, ctx):
        test_file = tmp_path / "utf8.txt"
        test_file.write_text("Hello unicode: \u00e9\u00e0\u00fc", encoding="utf-8")

        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)
        assert result.encoding == "utf-8"
        assert "\u00e9" in result.content

    async def test_non_utf8_fallback(self, tmp_path, ctx):
        test_file = tmp_path / "binary.txt"
        test_file.write_bytes(b"\x80\x81\x82\x83")

        tool = FileReaderTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)
        assert result.encoding == "latin-1"


class TestNoAllowedDirectories:
    async def test_no_restrictions_when_empty(self, tmp_path, ctx):
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        tool = FileReaderTool()  # No allowed_directories
        result = await tool.execute(FileReaderInput(file_path=str(test_file)), ctx)
        assert result.content == "content"


class TestToolProperties:
    def test_name(self):
        tool = FileReaderTool()
        assert tool.name == "file_reader"

    def test_tags(self):
        tool = FileReaderTool()
        assert "filesystem" in tool.tags
        assert "builtin" in tool.tags
