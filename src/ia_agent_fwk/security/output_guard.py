"""Output guardrails for LLM responses.

Ensures LLM responses don't leak system prompts, internal file names,
or other sensitive information before sending to the user.

Usage:
    guard = OutputGuard(blocked_terms=["system prompt", "AUTOS.pdf"])
    cleaned = guard.check_and_clean(response_text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default patterns to detect system prompt leakage
_SYSTEM_PROMPT_LEAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(my|the)\s+system\s+prompt\s+(is|says)", re.IGNORECASE),
    re.compile(r"(mi|el)\s+prompt\s+del?\s+sistema", re.IGNORECASE),
    re.compile(r"(my|the)\s+instructions?\s+(are|is|say)", re.IGNORECASE),
    re.compile(r"(mis|las)\s+instrucciones\s+(son|dicen)", re.IGNORECASE),
    re.compile(r"I\s+was\s+(told|instructed)\s+to", re.IGNORECASE),
    re.compile(r"me\s+(dijeron|instruyeron)\s+que", re.IGNORECASE),
]

# File name patterns to redact
_FILE_PATTERN = re.compile(r"\b[\w\-]+\.(pdf|docx|xlsx|json|txt|md)\b", re.IGNORECASE)


@dataclass
class OutputGuard:
    """Checks and cleans LLM output before sending to user.

    Parameters
    ----------
    blocked_terms:
        Exact strings to redact from output (case-insensitive).
    redact_filenames:
        Whether to redact file names (*.pdf, *.docx, etc).
    check_prompt_leak:
        Whether to check for system prompt leakage patterns.
    replacement:
        Text to replace blocked content with.

    """

    blocked_terms: list[str] = field(default_factory=list)
    redact_filenames: bool = True
    check_prompt_leak: bool = True
    replacement: str = ""

    def check_and_clean(self, text: str) -> str:
        """Clean LLM output, removing sensitive content."""
        cleaned = text

        # Redact blocked terms
        for term in self.blocked_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(cleaned):
                logger.info("Redacted blocked term from output: %s", term[:30])
                cleaned = pattern.sub(self.replacement, cleaned)

        # Redact file names
        if self.redact_filenames:
            matches = _FILE_PATTERN.findall(cleaned)
            if matches:
                logger.info("Redacted %d filename(s) from output", len(matches))
                cleaned = _FILE_PATTERN.sub(self.replacement, cleaned)

        # Check for system prompt leakage
        if self.check_prompt_leak:
            for pattern in _SYSTEM_PROMPT_LEAK_PATTERNS:
                if pattern.search(cleaned):
                    logger.warning("System prompt leak detected in output")
                    return "Lo siento, no puedo compartir esa información. ¿Puedo ayudarte con algo más?"

        return cleaned
