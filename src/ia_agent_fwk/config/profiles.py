"""Environment profile resolution for ia-agent-fwk configuration.

Determines which configuration profile to load based on the
``IAFWK_APP__ENVIRONMENT`` environment variable.
"""

from __future__ import annotations

import os

VALID_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "development",
        "testing",
        "staging",
        "production",
    }
)

DEFAULT_ENVIRONMENT: str = "development"

ENV_VAR_NAME: str = "IAFWK_APP__ENVIRONMENT"


def resolve_environment() -> str:
    """Resolve the active environment profile from env vars.

    Returns the value of ``IAFWK_APP__ENVIRONMENT`` if set and valid,
    otherwise falls back to ``"development"``.

    Raises
    ------
    ValueError
        If the environment variable is set to an unrecognised value.

    """
    env = os.environ.get(ENV_VAR_NAME, "").strip()
    if not env:
        return DEFAULT_ENVIRONMENT

    if env not in VALID_ENVIRONMENTS:
        valid_list = ", ".join(sorted(VALID_ENVIRONMENTS))
        msg = f"Invalid environment '{env}' (from {ENV_VAR_NAME}). Valid values: {valid_list}"
        raise ValueError(msg)

    return env
