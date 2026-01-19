"""Streaming module exception hierarchy."""

from __future__ import annotations


class StreamingError(Exception):
    """Base exception for streaming errors."""


class StreamConnectionError(StreamingError):
    """Raised when a streaming connection fails or is lost."""


class StreamTimeoutError(StreamingError):
    """Raised when a streaming operation times out."""
