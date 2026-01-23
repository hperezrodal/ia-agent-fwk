"""Plugin discovery mechanisms.

Provides two discovery strategies:

1. **Entry points** -- discovers plugins installed as Python packages
   that declare the ``ia_agent_fwk.plugins`` entry point group.
2. **Directory scanning** -- discovers plugins by importing Python
   files from a directory and finding ``Plugin`` subclasses.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING

from ia_agent_fwk.plugins.exceptions import PluginLoadError

if TYPE_CHECKING:
    from pathlib import Path

    from ia_agent_fwk.plugins.base import Plugin

logger = logging.getLogger(__name__)

_DEFAULT_ENTRY_POINT_GROUP = "ia_agent_fwk.plugins"


def discover_plugins_from_entry_points(
    group: str = _DEFAULT_ENTRY_POINT_GROUP,
) -> list[type[Plugin]]:
    """Discover plugin classes from installed package entry points.

    Parameters
    ----------
    group:
        The entry point group name to search.

    Returns
    -------
    list[type[Plugin]]
        List of discovered plugin classes.

    """
    from ia_agent_fwk.plugins.base import Plugin as PluginBase  # noqa: PLC0415

    discovered: list[type[Plugin]] = []
    entry_points = importlib.metadata.entry_points()

    # Python 3.12+ returns SelectableGroups; 3.9-3.11 returns dict
    eps = (
        entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, [])  # type: ignore[arg-type]
    )

    for ep in eps:
        try:
            plugin_cls = ep.load()
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load entry point '{ep.name}': {exc}"
            logger.warning(msg)
            continue

        if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, PluginBase) and plugin_cls is not PluginBase):
            msg = f"Entry point '{ep.name}' does not point to a Plugin subclass, skipping"
            logger.warning(msg)
            continue

        discovered.append(plugin_cls)

    return discovered


def discover_plugins_from_directory(directory: Path) -> list[type[Plugin]]:
    """Discover plugin classes by scanning Python files in a directory.

    Each ``.py`` file (excluding ``__init__.py``) in the directory is
    imported and inspected for concrete ``Plugin`` subclasses.

    Parameters
    ----------
    directory:
        The directory path to scan.

    Returns
    -------
    list[type[Plugin]]
        List of discovered plugin classes.

    Raises
    ------
    PluginLoadError
        If the directory does not exist.

    """
    from ia_agent_fwk.plugins.base import Plugin as PluginBase  # noqa: PLC0415

    if not directory.is_dir():
        msg = f"Plugin directory does not exist: {directory}"
        raise PluginLoadError(msg)

    discovered: list[type[Plugin]] = []

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"_ia_agent_fwk_plugin_{py_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to import plugin file '{py_file}': {exc}"
            logger.warning(msg)
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, PluginBase) and obj is not PluginBase and not inspect.isabstract(obj):
                discovered.append(obj)

    return discovered
