"""File reader built-in tool with path traversal prevention.

Reads file contents from sandboxed directories. Uses ``Path.resolve()``
and prefix checking to prevent path traversal attacks.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ia_agent_fwk.tools.base import Tool, ToolContext
from ia_agent_fwk.tools.exceptions import ToolExecutionError

# Default maximum file size: 10 MB
_DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024


class FileReaderInput(BaseModel):
    """Input schema for the file reader tool."""

    model_config = ConfigDict(frozen=True)

    file_path: str


class FileReaderOutput(BaseModel):
    """Output schema for the file reader tool."""

    model_config = ConfigDict(frozen=True)

    content: str
    file_size: int
    encoding: str


class FileReaderTool(Tool):
    """Read file contents from sandboxed directories.

    Parameters
    ----------
    allowed_directories:
        List of directory paths that the tool is allowed to read from.
        If empty, all paths are allowed (not recommended for production).
    max_file_size:
        Maximum file size in bytes. Default: 10 MB.

    """

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        max_file_size: int = _DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        self._allowed_directories = [Path(d).resolve() for d in (allowed_directories or [])]
        self._max_file_size = max_file_size

    @property
    def name(self) -> str:
        return "file_reader"

    @property
    def description(self) -> str:
        return "Read the contents of a file from allowed directories."

    @property
    def input_schema(self) -> type[BaseModel]:
        return FileReaderInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return FileReaderOutput

    @property
    def tags(self) -> list[str]:
        return ["filesystem", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Read the file contents."""
        assert isinstance(validated_input, FileReaderInput)  # noqa: S101
        file_path_str = validated_input.file_path

        # Resolve the path to an absolute canonical path
        try:
            resolved_path = Path(file_path_str).resolve(strict=True)  # noqa: ASYNC240
        except (OSError, ValueError) as exc:
            msg = f"File not found or inaccessible: {file_path_str} ({exc})"
            raise ToolExecutionError(msg, tool_name="file_reader") from exc

        # Check against allowed directories
        if self._allowed_directories:
            allowed = False
            for allowed_dir in self._allowed_directories:
                try:
                    resolved_path.relative_to(allowed_dir)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                msg = f"Access denied: '{file_path_str}' is outside allowed directories."
                raise ToolExecutionError(msg, tool_name="file_reader")

        # Check that it's a file
        if not resolved_path.is_file():
            msg = f"Path is not a file: {file_path_str}"
            raise ToolExecutionError(msg, tool_name="file_reader")

        # Check file size
        file_size = resolved_path.stat().st_size
        if file_size > self._max_file_size:
            msg = f"File too large: {file_size} bytes (maximum: {self._max_file_size} bytes)."
            raise ToolExecutionError(msg, tool_name="file_reader")

        # Read file content
        encoding = "utf-8"
        try:
            content = resolved_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to latin-1 which can decode any byte sequence
            encoding = "latin-1"
            content = resolved_path.read_text(encoding=encoding)
        except OSError as exc:
            msg = f"Failed to read file: {exc}"
            raise ToolExecutionError(msg, tool_name="file_reader") from exc

        return FileReaderOutput(
            content=content,
            file_size=file_size,
            encoding=encoding,
        )
