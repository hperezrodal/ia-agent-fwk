"""Streaming backpressure utility.

Provides a bounded-buffer wrapper for async iterators that drops the oldest
chunks when the consumer cannot keep up, as required by FR-024.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ia_agent_fwk.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default buffer size per FR-024.
DEFAULT_BUFFER_SIZE: int = 256


async def buffered_stream(
    source: AsyncIterator[T],
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> AsyncIterator[T]:
    """Wrap *source* with bounded backpressure buffering.

    When the internal ``asyncio.Queue`` is full, the oldest chunk is
    discarded and a warning is logged.  This prevents unbounded memory
    growth when the consumer is slower than the producer.

    Parameters
    ----------
    source:
        The upstream async iterator to buffer.
    buffer_size:
        Maximum number of items in the buffer (default: 256).

    """
    queue: asyncio.Queue[T | None] = asyncio.Queue(maxsize=buffer_size)
    dropped = 0
    chunks_yielded = 0
    collector = get_metrics_collector()

    async def _producer() -> None:
        nonlocal dropped
        try:
            async for item in source:
                if queue.full():
                    # Drop the oldest chunk.
                    try:
                        queue.get_nowait()
                        dropped += 1
                        collector.increment("llm_stream_chunks_dropped_total")
                        logger.warning(
                            "Streaming backpressure: dropped chunk (%d total drops)",
                            dropped,
                            extra={
                                "llm_data": {
                                    "event": "stream_chunk_dropped",
                                    "total_drops": dropped,
                                }
                            },
                        )
                    except asyncio.QueueEmpty:  # pragma: no cover
                        pass
                await queue.put(item)
        finally:
            # Sentinel to signal end of stream.
            await queue.put(None)

    task = asyncio.create_task(_producer())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            chunks_yielded += 1
            collector.increment("llm_stream_chunks_yielded_total")
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # Re-raise any exception from the producer.
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                raise exc
