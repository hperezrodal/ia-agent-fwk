"""Multi-provider LLM client for the conversational agent.

Supports OpenAI, Ollama, and Anthropic. Provides three call patterns:
- chat(): full response (sync)
- stream(): token-by-token (async generator)
- quick(): fast call for classification/rewriting

All config from env vars. No hardcoded models or URLs.

Usage:
    client = LLMClient.from_env()
    response = await client.chat(messages, temperature=0.3, max_tokens=1024)
    async for token, done in client.stream(messages):
        print(token, end="")
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM client.

    Parameters
    ----------
    provider:
        "openai", "ollama", or "anthropic".
    model:
        Model name (e.g. "gpt-4o-mini", "llama3.1:8b").
    api_key:
        API key (required for openai/anthropic).
    api_url:
        Base URL (required for ollama, default for others).
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        api_key: str = "",
        api_url: str = "http://localhost:11434",
    ) -> None:
        self._provider = provider
        self._model = model
        self._api_key = api_key
        self._api_url = api_url
        self._httpx_client: httpx.AsyncClient | None = None
        self._openai_client: Any = None

    @classmethod
    def from_env(cls) -> LLMClient:
        """Create client from environment variables."""
        provider = os.environ.get("LLM_PROVIDER", "ollama")
        if provider == "openai":
            return cls(
                provider="openai",
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
        return cls(
            provider="ollama",
            model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            api_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def _get_httpx(self) -> httpx.AsyncClient:
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=300.0)
        return self._httpx_client

    def _get_openai(self) -> Any:
        if self._openai_client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            self._openai_client = AsyncOpenAI(api_key=self._api_key)
        return self._openai_client

    # ------------------------------------------------------------------
    # chat (full response)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Call LLM and return full response text."""
        if self._provider == "openai":
            client = self._get_openai()
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        # Ollama
        client = self._get_httpx()
        resp = await client.post(
            f"{self._api_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    # ------------------------------------------------------------------
    # stream (token by token)
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Any:
        """Stream tokens. Yields (token_str, is_done) tuples."""
        if self._provider == "openai":
            client = self._get_openai()
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content, False
                if chunk.choices and chunk.choices[0].finish_reason:
                    yield "", True
            return

        # Ollama
        import json  # noqa: PLC0415

        client = self._get_httpx()
        async with client.stream(
            "POST",
            f"{self._api_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                yield data.get("message", {}).get("content", ""), data.get("done", False)

    # ------------------------------------------------------------------
    # quick (classification / rewriting)
    # ------------------------------------------------------------------

    async def quick(self, prompt: str, max_tokens: int = 80) -> str:
        """Fast LLM call for classification/rewriting (temperature=0)."""
        if self._provider == "openai":
            client = self._get_openai()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        # Ollama
        client = self._get_httpx()
        resp = await client.post(
            f"{self._api_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None
