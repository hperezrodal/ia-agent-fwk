"""Contextual Retrieval enrichment — LLM-generated context per chunk.

At ingestion time, for each chunk, sends the full document + chunk to an LLM
to generate a short context (~50-100 tokens) that situates the chunk within
the document. This context is prepended to the chunk content before embedding.

Based on: https://www.anthropic.com/news/contextual-retrieval

Usage:
    enricher = ContextualEnricher(
        ollama_url="http://localhost:11434",
        model="llama3.1:8b",
    )
    chunks = await enricher.enrich(chunks, full_document_text)
"""

from __future__ import annotations

import logging

import httpx

from ia_agent_fwk.ingestion.models import ProcessedChunk

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """\
<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a short succinct context (2-3 sentences) to situate this chunk \
within the overall document for improving search retrieval. Include: \
1) what company/organization the document belongs to, 2) what specific \
information this chunk contains, 3) keywords a user would search for \
to find this information. Only the context, do not repeat the chunk content."""


class ContextualEnricher:
    """Generate LLM context for each chunk and prepend it.

    Based on: https://www.anthropic.com/news/contextual-retrieval

    Parameters
    ----------
    ollama_url:
        Ollama API URL.
    model:
        LLM model name (e.g. "llama3.1:8b").
    max_doc_chars:
        Truncate document to this many chars (to fit in context window).
        llama3.1:8b has 128K context, so 100K chars is safe.

    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        max_doc_chars: int = 100_000,
        prompt_template: str | None = None,
    ) -> None:
        self._url = ollama_url
        self._model = model
        self._max_doc_chars = max_doc_chars
        self._prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._client = httpx.AsyncClient(timeout=120.0)

    async def enrich(
        self,
        chunks: list[ProcessedChunk],
        full_document: str,
    ) -> list[ProcessedChunk]:
        """Generate contextual prefix for each chunk and prepend it.

        Parameters
        ----------
        chunks:
            Chunks to enrich.
        full_document:
            Full parsed document text (markdown or plain).

        """
        doc_text = full_document[: self._max_doc_chars]
        total = len(chunks)
        enriched = 0

        for i, chunk in enumerate(chunks):
            # Skip very short chunks or table parents (too large for context)
            if chunk.metadata.get("table_role") == "parent":
                continue
            if len(chunk.content) < 30:
                continue

            try:
                context = await self._generate_context(doc_text, chunk.content)
                if context:
                    chunk.content = f"{context}\n\n{chunk.content}"
                    enriched += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Context generation failed for chunk %d: %s", i, exc)

            if (i + 1) % 10 == 0:
                logger.info("Contextual enrichment: %d/%d chunks", i + 1, total)

        logger.info(
            "Contextual enrichment done: %d/%d chunks enriched",
            enriched,
            total,
        )
        return chunks

    async def _generate_context(self, document: str, chunk: str) -> str:
        """Call Ollama to generate context for a chunk."""
        prompt = self._prompt_template.format(document=document, chunk=chunk)

        response = await self._client.post(
            f"{self._url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 150,
                },
            },
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
