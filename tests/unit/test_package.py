"""Tests for package import, version, and module structure."""

from __future__ import annotations

import importlib
import re

import pytest

import ia_agent_fwk

ALL_MODULES = [
    "config",
    "api",
    "agents",
    "llm",
    "tools",
    "memory",
    "rag",
    "orchestration",
    "execution",
    "streaming",
    "plugins",
    "integrations",
    "observability",
    "security",
    "db",
]


@pytest.mark.unit
class TestPackage:
    """Verify the ia_agent_fwk package is correctly installed."""

    def test_import_package(self):
        """Package can be imported."""
        assert ia_agent_fwk is not None

    def test_version_string(self):
        """Package exposes a valid semver-style version string."""
        assert hasattr(ia_agent_fwk, "__version__")
        assert isinstance(ia_agent_fwk.__version__, str)
        assert len(ia_agent_fwk.__version__) > 0
        # Basic semver pattern: major.minor.patch
        assert re.match(r"^\d+\.\d+\.\d+", ia_agent_fwk.__version__)

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_all_modules_importable(self, module_name: str):
        """Each of the 15 planned modules can be imported."""
        mod = importlib.import_module(f"ia_agent_fwk.{module_name}")
        assert mod is not None
