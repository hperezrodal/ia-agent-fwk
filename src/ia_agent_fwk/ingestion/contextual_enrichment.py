"""Contextual Retrieval enrichment — LLM-generated context per chunk.

At ingestion time, for each chunk, sends the full document + chunk to an LLM
to generate a short context (~50-100 tokens) that situates the chunk within
the document. This context is prepended to the chunk content before embedding.

Based on: https://www.anthropic.com/news/contextual-retrieval

Usage:
    # Ollama (default)
    enricher = ContextualEnricher(model="llama3.1:8b")

    # OpenAI
    enricher = ContextualEnricher(provider="openai", model="gpt-4o", api_key="sk-...")

    chunks = await enricher.enrich(chunks, full_document_text)
"""

from __future__ import annotations

import asyncio
import logging
import os

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
    provider:
        LLM provider: "ollama" or "openai".
    model:
        LLM model name (e.g. "llama3.1:8b", "gpt-4o").
    api_key:
        API key for OpenAI. Falls back to OPENAI_API_KEY env var.
    ollama_url:
        Ollama API URL (only used when provider="ollama").
    max_doc_chars:
        Truncate document to this many chars (to fit in context window).

    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str | None = None,
        ollama_url: str = "http://localhost:11434",
        max_doc_chars: int = 100_000,
        prompt_template: str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._ollama_url = ollama_url
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
                context = await self._generate_with_retry(doc_text, chunk.content)
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

    async def _generate_with_retry(
        self,
        document: str,
        chunk: str,
        max_retries: int = 3,
    ) -> str:
        """Call _generate_context with exponential backoff on rate limits."""
        for attempt in range(max_retries + 1):
            try:
                result = await self._generate_context(document, chunk)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.info(
                        "Rate limited, waiting %ds (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
            else:
                # Small delay between calls to avoid hitting rate limits
                await asyncio.sleep(0.5)
                return result
        return ""

    async def _generate_context(self, document: str, chunk: str) -> str:
        """Call LLM to generate context for a chunk."""
        prompt = self._prompt_template.format(document=document, chunk=chunk)

        if self._provider == "openai":
            return await self._generate_openai(prompt)
        return await self._generate_ollama(prompt)

    async def _generate_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        response = await self._client.post(
            f"{self._ollama_url}/api/generate",
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

    async def _generate_openai(self, prompt: str) -> str:
        """Call OpenAI Chat Completions API."""
        response = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 150,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
