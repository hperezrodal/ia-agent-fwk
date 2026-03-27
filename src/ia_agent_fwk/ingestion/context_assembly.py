"""Context assembly — format search results for LLM consumption.

Converts raw search results into a structured text context that the LLM
can use to answer questions. Format-agnostic and domain-agnostic.

Usage:
    from ia_agent_fwk.ingestion.context_assembly import assemble_context

    context = assemble_context(results, format="numbered")
    # Inject into LLM prompt as {context}
"""

from __future__ import annotations

from ia_agent_fwk.ingestion.hybrid_store import HybridSearchResult


def assemble_context(
    results: list[HybridSearchResult],
    *,
    format: str = "numbered",  # noqa: A002
    max_chars: int = 0,
    include_metadata: bool = True,
) -> str:
    """Assemble search results into a text context for LLM.

    Parameters
    ----------
    results:
        Search results to format.
    format:
        "numbered" — [1] source: section\\ncontent
        "xml" — <chunk index="1" source="..." section="...">content</chunk>
        "plain" — just the content, separated by blank lines
    max_chars:
        Truncate total context to this many chars. 0 = no limit.
    include_metadata:
        Include source/section/type labels in each chunk.

    """
    if not results:
        return ""

    formatters = {
        "numbered": _format_numbered,
        "xml": _format_xml,
        "plain": _format_plain,
    }
    formatter = formatters.get(format, _format_numbered)

    parts: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, 1):
        chunk_text = formatter(i, r, include_metadata)

        if max_chars > 0 and total_chars + len(chunk_text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                parts.append(chunk_text[:remaining] + "\n[...]")
            break

        parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n\n".join(parts)


def _format_numbered(
    index: int,
    r: HybridSearchResult,
    include_metadata: bool,
) -> str:
    """Format: [1] (source: file.pdf, section: Coberturas)\\ncontent"""
    if include_metadata:
        source = r.metadata.get("document_id", "")
        section = r.metadata.get("section", "")
        chunk_type = r.metadata.get("chunk_type", "text")
        table_role = r.metadata.get("table_role", "")
        if table_role:
            chunk_type = f"table:{table_role}"

        meta_parts = []
        if source:
            meta_parts.append(f"source: {source}")
        if section:
            meta_parts.append(f"section: {section}")
        meta_parts.append(f"type: {chunk_type}")

        header = f"[{index}] ({', '.join(meta_parts)})"
        return f"{header}\n{r.content}"
    return f"[{index}]\n{r.content}"


def _format_xml(
    index: int,
    r: HybridSearchResult,
    include_metadata: bool,
) -> str:
    """Format: <chunk index="1" source="..." section="...">content</chunk>"""
    if include_metadata:
        source = r.metadata.get("document_id", "")
        section = r.metadata.get("section", "")
        attrs = f'index="{index}"'
        if source:
            attrs += f' source="{source}"'
        if section:
            attrs += f' section="{section}"'
        return f"<chunk {attrs}>\n{r.content}\n</chunk>"
    return f'<chunk index="{index}">\n{r.content}\n</chunk>'


def _format_plain(
    index: int,  # noqa: ARG001
    r: HybridSearchResult,
    include_metadata: bool,  # noqa: ARG001
) -> str:
    """Format: just the content."""
    return r.content
