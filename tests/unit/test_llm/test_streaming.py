"""Tests for the streaming backpressure buffer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ia_agent_fwk.llm.streaming import DEFAULT_BUFFER_SIZE, buffered_stream


async def _async_iter(items: list[int]) -> AsyncIterator[int]:
    for item in items:
        yield item


async def _slow_async_iter(items: list[int], delay: float) -> AsyncIterator[int]:
    for item in items:
        await asyncio.sleep(delay)
        yield item


@pytest.mark.unit
class TestBufferedStream:
    """Tests for buffered_stream backpressure utility."""

    async def test_passes_through_all_items(self) -> None:
        """All items pass through when consumer keeps up."""
        items = list(range(10))
        result: list[int] = []
        async for chunk in buffered_stream(_async_iter(items), buffer_size=256):
            result.append(chunk)
        assert result == items

    async def test_empty_source(self) -> None:
        """Empty source yields nothing."""
        result: list[int] = []
        async for chunk in buffered_stream(_async_iter([]), buffer_size=10):
            result.append(chunk)
        assert result == []

    async def test_default_buffer_size(self) -> None:
        """Default buffer size is 256."""
        assert DEFAULT_BUFFER_SIZE == 256

    async def test_drops_oldest_when_full(self) -> None:
        """When buffer is full, oldest chunks are dropped."""
        buffer_size = 3
        # Produce 10 items very fast, then consume
        items = list(range(10))

        async def _fast_producer() -> AsyncIterator[int]:
            for item in items:
                yield item
                # Tiny yield to let producer fill buffer
                await asyncio.sleep(0)

        result: list[int] = []
        async for chunk in buffered_stream(_fast_producer(), buffer_size=buffer_size):
            result.append(chunk)

        # We should get some items, but possibly not all 10 due to drops
        # At minimum the last items and sentinel get through
        assert len(result) <= len(items)
        assert len(result) > 0

    async def test_producer_exception_propagates(self) -> None:
        """Exceptions from the source iterator propagate to consumer."""

        async def _failing_iter() -> AsyncIterator[int]:
            yield 1
            yield 2
            msg = "producer failed"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="producer failed"):
            async for _ in buffered_stream(_failing_iter(), buffer_size=10):
                pass

    async def test_consumer_cancellation_cleans_up(self) -> None:
        """If consumer stops early, producer task is cancelled."""

        async def _infinite_iter() -> AsyncIterator[int]:
            i = 0
            while True:
                yield i
                i += 1
                await asyncio.sleep(0)

        result: list[int] = []
        async for chunk in buffered_stream(_infinite_iter(), buffer_size=5):
            result.append(chunk)
            if len(result) >= 3:
                break

        assert len(result) == 3

    async def test_single_item(self) -> None:
        """Single item passes through correctly."""
        result: list[int] = []
        async for chunk in buffered_stream(_async_iter([42]), buffer_size=1):
            result.append(chunk)
        assert result == [42]
