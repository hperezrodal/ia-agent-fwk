"""SQL identifier validation utilities for PostgreSQL memory backends.

Prevents SQL injection by ensuring that table names, collection names,
and metadata filter keys conform to a safe identifier pattern.
"""

from __future__ import annotations

import re

from ia_agent_fwk.memory.exceptions import MemoryConfigError

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_sql_identifier(value: str, *, label: str = "identifier") -> str:
    """Validate that *value* is a safe SQL identifier.

    Only alphanumeric characters and underscores are allowed, and the
    identifier must start with a letter or underscore.  Maximum length
    is 63 characters (PostgreSQL's ``NAMEDATALEN - 1``).

    Parameters
    ----------
    value:
        The string to validate.
    label:
        Human-readable label for error messages (e.g. ``"table name"``).

    Returns
    -------
    str
        The validated identifier (unchanged).

    Raises
    ------
    MemoryConfigError
        If the identifier is invalid.

    """
    if not value:
        msg = f"Invalid {label}: must not be empty"
        raise MemoryConfigError(msg)

    if len(value) > 63:  # noqa: PLR2004
        msg = f"Invalid {label}: {value!r} exceeds 63 characters"
        raise MemoryConfigError(msg)

    if not _SAFE_IDENTIFIER_RE.match(value):
        msg = (
            f"Invalid {label}: {value!r} contains disallowed characters. "
            "Only alphanumeric characters and underscores are permitted, "
            "and the identifier must start with a letter or underscore."
        )
        raise MemoryConfigError(msg)

    return value
