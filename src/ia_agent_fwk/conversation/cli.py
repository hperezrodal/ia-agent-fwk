"""Interactive CLI chat client for any /chat/stream SSE endpoint.

Generic chat client that connects to a server exposing ``/chat/stream``
(SSE) or ``/chat`` (synchronous) endpoints.  Displays status indicators,
streaming tokens, source references, and timing information.

Usage::

    # Connect to a local chat API:
    python -m ia_agent_fwk.conversation.cli

    # Specify endpoint and agent:
    python -m ia_agent_fwk.conversation.cli --api-url http://localhost:8090 --agent my-agent

    # Disable streaming:
    python -m ia_agent_fwk.conversation.cli --no-stream
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_API_URL = "http://localhost:8090"
_DEFAULT_AGENT = "default"
_CLIENT_TIMEOUT = 300.0

# Human-readable labels for common status codes emitted by SSE endpoints.
_STATUS_LABELS: dict[str, str] = {
    "searching": "Searching...",
    "thinking": "Generating response...",
    "processing": "Processing...",
    "retrieving": "Retrieving context...",
}

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_CLEAR_LINE = "\r\033[K"


# ---------------------------------------------------------------------------
# Session context (avoids passing many positional args)
# ---------------------------------------------------------------------------


@dataclass
class _ChatContext:
    """Mutable state passed through the REPL loop."""

    client: httpx.Client
    api_url: str
    agent: str
    session_id: str | None
    use_stream: bool


# ---------------------------------------------------------------------------
# Chat transport functions
# ---------------------------------------------------------------------------


def _chat_stream(
    ctx: _ChatContext,
    message: str,
) -> dict[str, Any]:
    """Send *message* via the ``/chat/stream`` SSE endpoint and print tokens as they arrive.

    The SSE payload is expected to contain JSON objects with **one** of:

    * ``{"status": "...", "message": "..."}`` -- a status indicator
    * ``{"token": "..."}`` -- an incremental response token
    * ``{"done": true, ...}`` -- final metadata (session_id, sources, duration_ms, ...)

    Returns the final event data dict.  Also mutates ``ctx.session_id``.
    """
    with ctx.client.stream(
        "POST",
        f"{ctx.api_url}/chat/stream",
        json={"session_id": ctx.session_id, "message": message, "agent": ctx.agent},
    ) as response:
        response.raise_for_status()
        final_data: dict[str, Any] = {}
        first_token = True

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data: dict[str, Any] = json.loads(line[6:])

            if "status" in data:
                status = data["status"]
                label = _STATUS_LABELS.get(status, data.get("message", status))
                print(f"{_CLEAR_LINE}  {label}", end="", flush=True)  # noqa: T201

            elif "token" in data:
                if first_token:
                    print(_CLEAR_LINE, end="", flush=True)  # noqa: T201
                    first_token = False
                print(data["token"], end="", flush=True)  # noqa: T201

            elif data.get("done"):
                final_data = data

        print()  # noqa: T201  -- newline after streaming
        ctx.session_id = final_data.get("session_id", ctx.session_id)
        return final_data


def _chat_sync(
    ctx: _ChatContext,
    message: str,
) -> dict[str, Any]:
    """Send *message* via the synchronous ``/chat`` endpoint."""
    resp = ctx.client.post(
        f"{ctx.api_url}/chat",
        json={"session_id": ctx.session_id, "message": message, "agent": ctx.agent},
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    print(data.get("response", ""))  # noqa: T201
    ctx.session_id = data.get("session_id")
    return data


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the interactive chat REPL."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat client for /chat/stream SSE endpoints.",
    )
    parser.add_argument(
        "--api-url",
        default=_DEFAULT_API_URL,
        help=f"Base URL of the chat API (default: {_DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--agent",
        default=_DEFAULT_AGENT,
        help=f"Agent identifier to send in requests (default: {_DEFAULT_AGENT})",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Existing session ID to resume (default: new session)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Use synchronous /chat endpoint instead of SSE streaming",
    )
    args = parser.parse_args(argv)

    api_url: str = args.api_url.rstrip("/")
    use_stream: bool = not args.no_stream

    print(f"Chat CLI (agent: {args.agent}, stream: {'on' if use_stream else 'off'})")  # noqa: T201
    if args.session:
        print(f"Session: {args.session}")  # noqa: T201
    print("Type your message. 'exit'/'salir' to quit, 'new'/'nueva' for a new session.\n")  # noqa: T201

    ctx = _ChatContext(
        client=httpx.Client(timeout=_CLIENT_TIMEOUT),
        api_url=api_url,
        agent=args.agent,
        session_id=args.session,
        use_stream=use_stream,
    )

    try:
        _repl_loop(ctx)
    finally:
        ctx.client.close()


def _repl_loop(ctx: _ChatContext) -> None:
    """Inner REPL loop, separated for readability."""
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")  # noqa: T201
            break

        if not user_input:
            continue

        if user_input.lower() in ("salir", "exit", "quit"):
            print("Bye!")  # noqa: T201
            break

        if user_input.lower() in ("nueva", "new", "reset"):
            if ctx.session_id:
                try:
                    ctx.client.delete(f"{ctx.api_url}/sessions/{ctx.session_id}")
                except httpx.HTTPError:
                    pass  # best-effort cleanup
            ctx.session_id = None
            print("--- New conversation ---\n")  # noqa: T201
            continue

        print("\nBot: ", end="", flush=True)  # noqa: T201

        try:
            if ctx.use_stream:
                final = _chat_stream(ctx, user_input)
            else:
                final = _chat_sync(ctx, user_input)
            sources: list[dict[str, Any]] = final.get("sources", [])
            duration: float = final.get("duration_ms", 0)
        except httpx.ConnectError:
            print(f"\nError: could not connect to the API at {ctx.api_url}.")  # noqa: T201
            print("  Make sure the server is running.")  # noqa: T201
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}")  # noqa: T201
            continue

        # -- metadata --
        if sources:
            src_names = ", ".join(s.get("document", "?") for s in sources[:3])
            print(f"  [sources: {src_names}]")  # noqa: T201
        print(f"  [{duration:.0f}ms | session: {ctx.session_id}]\n")  # noqa: T201


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
