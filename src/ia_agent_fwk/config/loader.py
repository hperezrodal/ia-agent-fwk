"""Configuration loader for ia-agent-fwk.

Implements the four-layer configuration precedence:
  1. Environment variables (``IAFWK_*``)  -- highest
  2. Environment-specific YAML (``config/{environment}.yaml``)
  3. Default YAML (``config/default.yaml``)
  4. Pydantic Settings defaults                        -- lowest
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ia_agent_fwk.config.profiles import resolve_environment
from ia_agent_fwk.config.settings import AppSettings


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*.

    - Dict values are merged recursively.
    - All other values in *override* replace the corresponding value in *base*.
    - Keys present only in *base* are preserved.

    Returns a **new** dict; neither input is mutated.
    """
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def _load_yaml_data(
    environment: str,
    config_dir: Path,
) -> dict[str, Any]:
    """Load and merge YAML config files, returning a merged dict.

    Steps:
      1. Load ``config/default.yaml`` (if it exists).
      2. Deep-merge ``config/{environment}.yaml`` on top (if it exists).
    """
    config_data: dict[str, Any] = {}

    default_path = config_dir / "default.yaml"
    if default_path.exists():
        with default_path.open() as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                config_data = raw

    env_path = config_dir / f"{environment}.yaml"
    if env_path.exists():
        with env_path.open() as f:
            env_data = yaml.safe_load(f)
            if isinstance(env_data, dict):
                config_data = deep_merge(config_data, env_data)

    return config_data


def _remove_env_overridden_keys(
    data: dict[str, Any],
    prefix: str,
    delimiter: str,
    path: str = "",
) -> dict[str, Any]:
    """Remove keys from *data* that have corresponding env vars set.

    For each leaf key, checks if an env var like ``IAFWK_<PATH>__<KEY>``
    exists.  If so, the key is removed so that Pydantic Settings can read
    the env var without being overridden by the init kwarg.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        env_key = f"{prefix}{path}{key}".upper()
        if isinstance(value, dict):
            nested = _remove_env_overridden_keys(
                value,
                prefix=prefix,
                delimiter=delimiter,
                path=f"{path}{key}{delimiter}",
            )
            if nested:
                result[key] = nested
        elif env_key in os.environ:
            continue
        else:
            result[key] = value
    return result


def _validate_semantics(settings: AppSettings) -> None:
    """Check business-logic constraints that go beyond Pydantic type validation.

    Raises
    ------
    ValueError
        If a semantic constraint is violated.

    """
    errors: list[str] = []

    if settings.app.environment == "production":
        if settings.app.debug:
            errors.append("app.debug must be False in production")

        if not settings.auth.enabled:
            errors.append("auth.enabled must be True in production")

        if settings.auth.jwt.enabled and not settings.auth.jwt.secret_key:
            errors.append(
                "auth.jwt.secret_key must be set when JWT is enabled in production (set IAFWK_AUTH__JWT__SECRET_KEY)"
            )

    # Verify at least one LLM provider is configured (has a non-empty api_key
    # or a base_url for local providers like Ollama).
    if settings.llm.providers:
        has_configured_provider = any(
            p.api_key.get_secret_value() or p.base_url for p in settings.llm.providers.values()
        )
        if not has_configured_provider and settings.app.environment == "production":
            errors.append("At least one LLM provider must have an api_key or base_url in production")

    if errors:
        msg = "Semantic validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)


def load_config(
    environment: str | None = None,
    config_dir: str | Path | None = None,
) -> AppSettings:
    """Load and validate application configuration.

    Parameters
    ----------
    environment:
        The environment profile name (e.g. ``"development"``).
        If *None*, the profile is resolved from the
        ``IAFWK_APP__ENVIRONMENT`` env var (default: ``"development"``).
    config_dir:
        Path to the directory containing YAML config files.
        Defaults to ``config/`` relative to the current working directory.

    Returns
    -------
    AppSettings
        A fully validated, frozen settings instance.

    Raises
    ------
    pydantic.ValidationError
        If configuration validation fails.
    ValueError
        If semantic validation fails (e.g. debug enabled in production).

    """
    if environment is None:
        environment = resolve_environment()

    config_dir_path = Path("config") if config_dir is None else Path(config_dir)

    # Load merged YAML data (default + profile)
    yaml_data = _load_yaml_data(environment, config_dir_path)

    # Build settings with correct precedence.
    # Pydantic Settings treats init kwargs as highest priority (above env vars).
    # To ensure env vars > YAML > defaults, we remove any YAML keys that have
    # a corresponding env var set, then pass the rest as init kwargs.
    filtered_yaml = _remove_env_overridden_keys(yaml_data, prefix="IAFWK_", delimiter="__")

    try:
        settings = AppSettings(**filtered_yaml)
    except ValidationError as exc:
        print(  # noqa: T201
            f"Configuration validation failed:\n{exc}",
            file=sys.stderr,
        )
        raise

    _validate_semantics(settings)
    return settings
