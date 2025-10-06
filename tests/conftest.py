"""Shared test fixtures for ia-agent-fwk."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


# ==============================================================================
# Config directory fixtures
# ==============================================================================


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with default.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    default_yaml = config_dir / "default.yaml"
    default_yaml.write_text(
        textwrap.dedent("""\
            app:
              name: "test-app"
              version: "0.0.1"
              environment: "development"
              debug: false
              log_level: "INFO"

            server:
              host: "0.0.0.0"
              port: 8000
              workers: 1
              reload: false

            database:
              url: "postgresql+asyncpg://postgres:postgres@localhost:5432/test_db"
              pool_size: 5
              max_overflow: 10
              echo: false

            redis:
              url: "redis://localhost:6379/0"
              max_connections: 10

            memory:
              default_backend: "in_memory"
        """),
    )

    return config_dir


@pytest.fixture
def tmp_config_dir_with_profiles(tmp_config_dir: Path) -> Path:
    """Extend tmp_config_dir with development and testing profile files."""
    development_yaml = tmp_config_dir / "development.yaml"
    development_yaml.write_text(
        textwrap.dedent("""\
            app:
              debug: true
              log_level: "DEBUG"

            server:
              reload: true
        """),
    )

    testing_yaml = tmp_config_dir / "testing.yaml"
    testing_yaml.write_text(
        textwrap.dedent("""\
            app:
              environment: "testing"
              debug: false
              log_level: "WARNING"

            memory:
              default_backend: "in_memory"
        """),
    )

    return tmp_config_dir
