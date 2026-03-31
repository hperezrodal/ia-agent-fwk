"""Input validation and prompt injection detection for LLM-facing APIs.

Validates user messages before they reach the LLM pipeline.
Detects common prompt injection patterns and enforces size limits.

Usage:
    guard = InputGuard(max_length=500)
    result = guard.check("normal question")       # InputCheckResult(ok=True)
    result = guard.check("ignore your instructions")  # InputCheckResult(ok=False, reason="prompt_injection")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Prompt injection patterns — language-agnostic + Spanish
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # English
    re.compile(r"ignore\s+(your|all|previous)\s+(instructions|rules|prompt)", re.IGNORECASE),
    re.compile(r"forget\s+(your|all|previous)\s+(instructions|rules|prompt)", re.IGNORECASE),
    re.compile(r"disregard\s+(your|all|previous)", re.IGNORECASE),
    re.compile(r"(show|print|reveal|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if|a|an)", re.IGNORECASE),
    re.compile(r"pretend\s+(you|to\s+be)", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"override\s+(system|prompt|instructions)", re.IGNORECASE),
    re.compile(r"\bDAN\b.*\bjailbreak\b", re.IGNORECASE),
    # Spanish
    re.compile(r"ignor[áa]\s+(tus|las|todas)\s+(instrucciones|reglas)", re.IGNORECASE),
    re.compile(r"olvidate\s+de\s+(tus|las)\s+(instrucciones|reglas)", re.IGNORECASE),
    re.compile(r"(mostr[áa]|decime|revel[áa])\s+(tu|el)\s+(prompt|system\s+prompt|instrucciones)", re.IGNORECASE),
    re.compile(r"ahora\s+sos\s+(un|una)", re.IGNORECASE),
    re.compile(r"nuevas?\s+instrucciones:", re.IGNORECASE),
    re.compile(r"(hac[ée]\s+de\s+cuenta|fing[ií]\s+que)", re.IGNORECASE),
]

# Patterns that indicate code/script injection attempts
_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"\{\{.*\}\}"),  # template injection
    re.compile(r"\$\{.*\}"),  # template literal injection
]


@dataclass(frozen=True)
class InputCheckResult:
    """Result of an input validation check."""

    ok: bool
    reason: str = ""
    detail: str = ""


class InputGuard:
    """Validates and sanitizes user input before it reaches the LLM.

    Parameters
    ----------
    max_length:
        Maximum allowed message length in characters.
    check_injection:
        Whether to check for prompt injection patterns.
    check_code:
        Whether to check for code/script injection.
    extra_patterns:
        Additional regex patterns to flag as injection.

    """

    def __init__(
        self,
        *,
        max_length: int = 500,
        check_injection: bool = True,
        check_code: bool = True,
        extra_patterns: list[re.Pattern[str]] | None = None,
    ) -> None:
        self._max_length = max_length
        self._check_injection = check_injection
        self._check_code = check_code
        self._patterns = list(_INJECTION_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def check(self, message: str) -> InputCheckResult:
        """Validate a user message.

        Returns InputCheckResult with ok=True if the message passes
        all checks, or ok=False with the reason for rejection.
        """
        # Empty check
        if not message or not message.strip():
            return InputCheckResult(ok=False, reason="empty", detail="Message is empty")

        # Length check
        if len(message) > self._max_length:
            logger.info(
                "Input rejected: too long (%d > %d)",
                len(message),
                self._max_length,
            )
            return InputCheckResult(
                ok=False,
                reason="too_long",
                detail=f"Message exceeds {self._max_length} characters",
            )

        # Prompt injection check
        if self._check_injection:
            for pattern in self._patterns:
                if pattern.search(message):
                    logger.warning(
                        "Prompt injection detected: pattern=%s, input=%s",
                        pattern.pattern[:50],
                        message[:80],
                    )
                    return InputCheckResult(
                        ok=False,
                        reason="prompt_injection",
                        detail="Message contains prohibited patterns",
                    )

        # Code injection check
        if self._check_code:
            for pattern in _CODE_PATTERNS:
                if pattern.search(message):
                    logger.warning("Code injection detected: %s", message[:80])
                    return InputCheckResult(
                        ok=False,
                        reason="code_injection",
                        detail="Message contains prohibited code patterns",
                    )

        return InputCheckResult(ok=True)

    def sanitize(self, message: str) -> str:
        """Strip potentially dangerous content from a message.

        Unlike check(), this doesn't reject — it cleans and returns.
        Use for messages that passed check() but need extra cleaning.
        """
        # Strip HTML tags
        cleaned = re.sub(r"<[^>]+>", "", message)
        # Strip control characters (keep newlines)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
        # Collapse excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
