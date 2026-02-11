"""Document loader tools for reading files from a documents directory.

Provides ``ListDocumentsTool`` to list available documents and
``LoadDocumentTool`` to read document content using the RAG loader
registry (supports .txt, .md, .html, .pdf).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ia_agent_fwk.tools.base import Tool
from ia_agent_fwk.tools.exceptions import ToolExecutionError

if TYPE_CHECKING:
    from ia_agent_fwk.tools.base import ToolContext

# Default documents directory — overridden via DOCUMENTS_DIR env var
_DEFAULT_DOCUMENTS_DIR = "/app/documents"
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}


def _get_documents_dir() -> Path:
    """Return the resolved documents directory path."""
    return Path(os.environ.get("DOCUMENTS_DIR", _DEFAULT_DOCUMENTS_DIR)).resolve()


# ---------------------------------------------------------------------------
# ListDocumentsTool
# ---------------------------------------------------------------------------


class ListDocumentsInput(BaseModel):
    """Input schema for the list documents tool."""

    model_config = ConfigDict(frozen=True)

    pattern: str = Field(
        default="*",
        description="Glob pattern to filter files (e.g. '*.pdf', '*.txt'). Default: all files.",
    )


class DocumentInfo(BaseModel):
    """Metadata about a single document."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(description="Name of the file.")
    size_bytes: int = Field(description="File size in bytes.")
    extension: str = Field(description="File extension (e.g. '.pdf').")


class ListDocumentsOutput(BaseModel):
    """Output schema for the list documents tool."""

    model_config = ConfigDict(frozen=True)

    documents: list[DocumentInfo] = Field(description="List of available documents.")
    total: int = Field(description="Total number of documents found.")
    directory: str = Field(description="Path to the documents directory.")


class ListDocumentsTool(Tool):
    """List available documents in the documents directory.

    Scans the configured documents directory for supported file types
    (.txt, .md, .html, .pdf) and returns their names and sizes.
    """

    @property
    def name(self) -> str:
        return "list_documents"

    @property
    def description(self) -> str:
        return "List all available documents in the documents directory. Returns filenames, sizes, and extensions."

    @property
    def input_schema(self) -> type[BaseModel]:
        return ListDocumentsInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ListDocumentsOutput

    @property
    def tags(self) -> list[str]:
        return ["document", "filesystem", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """List documents in the documents directory."""
        assert isinstance(validated_input, ListDocumentsInput)  # noqa: S101

        docs_dir = _get_documents_dir()
        if not docs_dir.is_dir():
            msg = f"Documents directory not found: {docs_dir}"
            raise ToolExecutionError(msg, tool_name="list_documents")

        documents: list[DocumentInfo] = []
        for path in sorted(docs_dir.glob(validated_input.pattern)):
            if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                documents.append(
                    DocumentInfo(
                        filename=path.name,
                        size_bytes=path.stat().st_size,
                        extension=path.suffix.lower(),
                    )
                )

        return ListDocumentsOutput(
            documents=documents,
            total=len(documents),
            directory=str(docs_dir),
        )


# ---------------------------------------------------------------------------
# LoadDocumentTool
# ---------------------------------------------------------------------------


class LoadDocumentInput(BaseModel):
    """Input schema for the load document tool."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(description="Name of the file to load (e.g. 'contract.pdf').")


class LoadDocumentOutput(BaseModel):
    """Output schema for the load document tool."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(description="Name of the loaded file.")
    content: str = Field(description="Extracted text content of the document.")
    doc_type: str = Field(description="Document type (text, markdown, html, pdf).")
    char_count: int = Field(description="Number of characters in the extracted content.")
    metadata: dict[str, str] = Field(default_factory=dict, description="File metadata.")


class LoadDocumentTool(Tool):
    """Load and extract text from a document in the documents directory.

    Uses the RAG ``LoaderRegistry`` to support multiple file formats
    (.txt, .md, .html, .pdf).  Path traversal is prevented by resolving
    the path and checking it stays within the documents directory.
    """

    @property
    def name(self) -> str:
        return "load_document"

    @property
    def description(self) -> str:
        return (
            "Load a document by filename from the documents directory and extract its text content. "
            "Supports .txt, .md, .html, and .pdf files."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        return LoadDocumentInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return LoadDocumentOutput

    @property
    def tags(self) -> list[str]:
        return ["document", "filesystem", "builtin"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:  # noqa: ARG002
        """Load and extract text from the specified document."""
        assert isinstance(validated_input, LoadDocumentInput)  # noqa: S101

        docs_dir = _get_documents_dir()
        if not docs_dir.is_dir():
            msg = f"Documents directory not found: {docs_dir}"
            raise ToolExecutionError(msg, tool_name="load_document")

        # Resolve and validate path (prevent traversal)
        file_path = (docs_dir / validated_input.filename).resolve()
        try:
            file_path.relative_to(docs_dir)
        except ValueError:
            msg = f"Access denied: '{validated_input.filename}' is outside the documents directory."
            raise ToolExecutionError(msg, tool_name="load_document")  # noqa: B904

        if not file_path.is_file():
            msg = f"Document not found: '{validated_input.filename}'"
            raise ToolExecutionError(msg, tool_name="load_document")

        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            msg = f"Unsupported file type: '{file_path.suffix}'. Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            raise ToolExecutionError(msg, tool_name="load_document")

        # Use RAG LoaderRegistry to load the document
        from ia_agent_fwk.rag.loaders.registry import LoaderRegistry  # noqa: PLC0415

        registry = LoaderRegistry()
        try:
            document = await registry.load(file_path)
        except Exception as exc:
            msg = f"Failed to load document '{validated_input.filename}': {exc}"
            raise ToolExecutionError(msg, tool_name="load_document") from exc

        metadata = {str(k): str(v) for k, v in document.metadata.items()}

        return LoadDocumentOutput(
            filename=validated_input.filename,
            content=document.content,
            doc_type=document.doc_type,
            char_count=len(document.content),
            metadata=metadata,
        )
