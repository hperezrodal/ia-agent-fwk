"""Query expansion — enrich queries with synonyms and domain terms.

The framework provides the base class and amount normalization.
Projects provide their own synonym dictionaries.

Usage:
    # With custom synonyms (project-level):
    expander = QueryExpander(synonyms={"agencia": ["sucursal", "oficina"]})
    expanded = expander.expand("que agencias tiene?")

    # Using the module-level function with default (empty) synonyms:
    from ia_agent_fwk.ingestion.query_expansion import expand_query
    expanded = expand_query("query text")
"""

from __future__ import annotations

import re


class QueryExpander:
    """Expand queries with synonyms and term normalization.

    Parameters
    ----------
    synonyms:
        Dictionary mapping terms to lists of synonyms.
        Override with domain-specific terms in your project.
    amount_patterns:
        List of (regex, replacement) tuples for amount normalization.

    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        amount_patterns: list[tuple[re.Pattern[str], str]] | None = None,
    ) -> None:
        self._synonyms = synonyms or {}
        self._amount_patterns = amount_patterns or [
            (re.compile(r"(\d+)\s*millon(?:es)?", re.IGNORECASE), r"$\1.000.000"),
            (re.compile(r"(\d+)\s*mil\b", re.IGNORECASE), r"$\1.000"),
        ]

    def expand(self, query: str) -> str:
        """Expand a query with synonyms and amount normalization."""
        expanded = query
        query_lower = query.lower()

        additions: list[str] = []
        for term, syns in self._synonyms.items():
            if term in query_lower:
                additions.extend(syns)

        for pattern, replacement in self._amount_patterns:
            expanded = pattern.sub(replacement, expanded)

        if additions:
            expanded = f"{expanded} {' '.join(additions)}"

        return expanded


# Default instance with no synonyms (projects override)
_default_expander = QueryExpander()


def expand_query(query: str) -> str:
    """Expand a query using the default (empty) expander."""
    return _default_expander.expand(query)
