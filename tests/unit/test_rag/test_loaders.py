"""Tests for document loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ia_agent_fwk.rag.exceptions import DocumentLoadError
from ia_agent_fwk.rag.loaders.markdown import MarkdownLoader
from ia_agent_fwk.rag.loaders.pdf import PDFLoader
from ia_agent_fwk.rag.loaders.registry import LoaderRegistry
from ia_agent_fwk.rag.loaders.text import TextLoader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestTextLoader:
    async def test_text_loader_loads_file(self, tmp_text_file: Path):
        loader = TextLoader()
        doc = await loader.load(tmp_text_file)
        assert "sample text content" in doc.content
        assert doc.doc_type == "text"
        assert doc.metadata["filename"] == "sample.txt"

    async def test_text_loader_file_not_found(self, tmp_path: Path):
        loader = TextLoader()
        with pytest.raises(DocumentLoadError, match="File not found"):
            await loader.load(tmp_path / "nonexistent.txt")

    async def test_can_load(self, tmp_text_file: Path):
        loader = TextLoader()
        assert loader.can_load(tmp_text_file)
        assert not loader.can_load("file.pdf")


@pytest.mark.unit
class TestMarkdownLoader:
    async def test_markdown_loader(self, tmp_md_file: Path):
        loader = MarkdownLoader()
        doc = await loader.load(tmp_md_file)
        assert "Heading" in doc.content
        assert "bold" in doc.content
        assert "italic" in doc.content
        # Markdown formatting should be stripped
        assert "**" not in doc.content
        assert "*italic*" not in doc.content
        assert doc.doc_type == "markdown"


@pytest.mark.unit
class TestPDFLoader:
    """F-016: PDFLoader unit tests."""

    @pytest.fixture(autouse=True)
    def _require_pypdf(self):
        pytest.importorskip("pypdf", reason="pypdf not installed")

    async def test_pdf_loader_loads_valid_pdf(self, tmp_path: Path):
        pdf_path = tmp_path / "test.pdf"

        # Build a minimal PDF with text using reportlab-free raw PDF stream
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000282 00000 n \n"
            b"0000000380 00000 n \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
            b"startxref\n457\n%%EOF\n"
        )
        pdf_path.write_bytes(pdf_bytes)

        loader = PDFLoader()
        doc = await loader.load(pdf_path)

        assert doc.doc_type == "pdf"
        assert "Hello PDF" in doc.content
        assert doc.source == str(pdf_path)

    async def test_pdf_loader_file_not_found(self, tmp_path: Path):
        loader = PDFLoader()
        with pytest.raises(DocumentLoadError, match="File not found"):
            await loader.load(tmp_path / "nonexistent.pdf")

    async def test_pdf_loader_metadata_includes_page_count_and_file_size(self, tmp_path: Path):
        pdf_path = tmp_path / "meta_test.pdf"
        # Two-page PDF
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 47 >>\nstream\nBT /F1 12 Tf 100 700 Td (Page One) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"6 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 7 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"7 0 obj\n<< /Length 47 >>\nstream\nBT /F1 12 Tf 100 700 Td (Page Two) Tj ET\n"
            b"endstream\nendobj\n"
            b"xref\n0 8\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000125 00000 n \n"
            b"0000000292 00000 n \n"
            b"0000000393 00000 n \n"
            b"0000000460 00000 n \n"
            b"0000000627 00000 n \n"
            b"trailer\n<< /Size 8 /Root 1 0 R >>\n"
            b"startxref\n728\n%%EOF\n"
        )
        pdf_path.write_bytes(pdf_bytes)

        loader = PDFLoader()
        doc = await loader.load(pdf_path)

        assert doc.metadata["page_count"] == 2
        assert doc.metadata["file_size_bytes"] > 0
        assert doc.metadata["filename"] == "meta_test.pdf"

    async def test_can_load(self):
        loader = PDFLoader()
        assert loader.can_load("document.pdf")
        assert loader.can_load("file.PDF")
        assert not loader.can_load("file.txt")


@pytest.mark.unit
class TestHTMLLoader:
    @pytest.fixture(autouse=True)
    def _require_bs4(self):
        pytest.importorskip("bs4", reason="beautifulsoup4 not installed")

    async def test_html_loader_extracts_text(self, tmp_html_file: Path):
        from ia_agent_fwk.rag.loaders.html import HTMLLoader

        loader = HTMLLoader()
        doc = await loader.load(tmp_html_file)
        assert "Hello" in doc.content
        assert "World" in doc.content
        # HTML tags should be stripped
        assert "<h1>" not in doc.content
        assert doc.doc_type == "html"


@pytest.mark.unit
class TestDOCXLoader:
    async def test_docx_loader_loads_file(self, tmp_docx_file: Path):
        from ia_agent_fwk.rag.loaders.docx import DOCXLoader

        loader = DOCXLoader()
        doc = await loader.load(tmp_docx_file)
        assert "First paragraph" in doc.content
        assert "Second paragraph" in doc.content
        assert doc.doc_type == "docx"
        assert doc.metadata["filename"] == "sample.docx"
        assert doc.metadata["paragraph_count"] == 2
        assert doc.metadata["file_size_bytes"] > 0

    async def test_docx_loader_file_not_found(self, tmp_path: Path):
        pytest.importorskip("docx", reason="python-docx not installed")
        from ia_agent_fwk.rag.loaders.docx import DOCXLoader

        loader = DOCXLoader()
        with pytest.raises(DocumentLoadError, match="File not found"):
            await loader.load(tmp_path / "nonexistent.docx")

    async def test_can_load(self):
        from ia_agent_fwk.rag.loaders.docx import DOCXLoader

        loader = DOCXLoader()
        assert loader.can_load("document.docx")
        assert not loader.can_load("file.txt")


@pytest.mark.unit
class TestLoaderRegistry:
    async def test_loader_registry_get_loader(self):
        registry = LoaderRegistry()
        loader = registry.get_loader("test.txt")
        assert isinstance(loader, TextLoader)

    async def test_loader_registry_unknown_extension(self):
        registry = LoaderRegistry()
        with pytest.raises(DocumentLoadError, match="No loader registered"):
            registry.get_loader("test.xyz")

    async def test_loader_registry_load(self, tmp_text_file: Path):
        registry = LoaderRegistry()
        doc = await registry.load(tmp_text_file)
        assert "sample text content" in doc.content

    async def test_register_custom_loader(self):
        registry = LoaderRegistry()
        registry.register(".txt", TextLoader())
        loader = registry.get_loader("my.txt")
        assert isinstance(loader, TextLoader)

    async def test_docx_loader_registered(self):
        from ia_agent_fwk.rag.loaders.docx import DOCXLoader

        registry = LoaderRegistry()
        loader = registry.get_loader("document.docx")
        assert isinstance(loader, DOCXLoader)
