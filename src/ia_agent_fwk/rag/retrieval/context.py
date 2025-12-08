"""Context assembler for formatting retrieved chunks into prompt context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ia_agent_fwk.observability.metrics import get_metrics_collector

if TYPE_CHECKING:
    from collections.abc import Callable

    from ia_agent_fwk.rag.models import RetrievalResult

_FORMAT_TEMPLATES: dict[str, str] = {
    "numbered": "[{index}] (source: {source}, score: {score:.2f}):\n{content}",
    "xml": '<chunk index="{index}" source="{source}" score="{score:.2f}">\n{content}\n</chunk>',
    "plain": "{content}",
}

_DEFAULT_FORMAT = "numbered"


class ContextAssembler:
    """Assemble retrieved chunks into a formatted context string.

    Parameters
    ----------
    template:
        Explicit format string for each chunk entry.  Available placeholders:
        ``{index}``, ``{source}``, ``{score}``, ``{content}``.
        When provided, it takes precedence over *format*.
    format:
        Name of a built-in format preset (``"numbered"``, ``"xml"``,
        ``"plain"``).  Ignored when *template* is given.  Defaults to
        ``"numbered"``.
    max_tokens:
        Optional upper bound on the total assembled context length
        (measured by *token_counter*).  Chunks that would cause the
        running total to exceed this budget are skipped.
    token_counter:
        Callable that returns the token count of a string.  When
        ``None`` and *max_tokens* is set, ``len`` (character count)
        is used as a fallback estimator.

    """

    def __init__(
        self,
        template: str | None = None,
        *,
        format: str = _DEFAULT_FORMAT,  # noqa: A002
        max_tokens: int | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if template is not None:
            self._template = template
        else:
            if format not in _FORMAT_TEMPLATES:
                msg = f"Unknown format preset {format!r}. Choose from {sorted(_FORMAT_TEMPLATES)}."
                raise ValueError(msg)
            self._template = _FORMAT_TEMPLATES[format]

        self._max_tokens = max_tokens
        self._token_counter: Callable[[str], int] = token_counter or len

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def assemble(self, chunks: list[RetrievalResult], template: str | None = None) -> str:
        """Format *chunks* into a context string.

        Parameters
        ----------
        chunks:
            Retrieval results ordered by relevance.
        template:
            Optional per-call override for the template.

        Returns
        -------
        str
            Formatted context string, or an empty string when *chunks*
            is empty.

        """
        if not chunks:
            return ""

        collector = get_metrics_collector()
        tpl = template or self._template
        parts: list[str] = []
        total_tokens = 0
        separator = "\n\n"
        skipped = 0

        for i, result in enumerate(chunks, start=1):
            rendered = tpl.format(
                index=i,
                source=result.chunk.source,
                score=result.score,
                content=result.chunk.content,
            )

            if self._max_tokens is not None:
                # Account for separator between parts (except the first).
                sep_cost = self._token_counter(separator) if parts else 0
                chunk_cost = self._token_counter(rendered)
                if total_tokens + sep_cost + chunk_cost > self._max_tokens:
                    skipped += 1
                    continue
                total_tokens += sep_cost + chunk_cost

            parts.append(rendered)

        assembled = separator.join(parts)
        collector.increment("rag_context_assembly_total")
        collector.observe("rag_context_chunks_included", len(parts))
        if skipped > 0:
            collector.observe("rag_context_chunks_skipped", skipped)
        collector.observe("rag_context_length_chars", len(assembled))
        return assembled
