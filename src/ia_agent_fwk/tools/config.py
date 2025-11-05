"""Tool system configuration models.

``ToolsConfig`` defines global tool settings.
``ToolPermissionConfig`` defines per-agent permission settings.
Both are ``BaseModel`` subclasses nested under ``AppSettings``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ToolPermissionConfig(BaseModel):
    """Per-agent tool permission configuration.

    Attributes
    ----------
    mode:
        Permission mode: ``allow_all``, ``allow_list``, ``deny_list``,
        or ``require_confirmation``.
    allowed:
        Tool names allowed when mode is ``allow_list``.
    denied:
        Tool names denied when mode is ``deny_list``.
    require_confirmation:
        Tool names requiring confirmation when mode is ``require_confirmation``.

    """

    model_config = ConfigDict(frozen=True)

    mode: str = "allow_all"
    allowed: list[str] = Field(default_factory=list)
    denied: list[str] = Field(default_factory=list)
    require_confirmation: list[str] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    """Global tool system configuration.

    Attributes
    ----------
    default_timeout:
        Default tool execution timeout in seconds.
    default_permission_mode:
        Default permission mode when no per-agent config exists.
    builtin_tools_enabled:
        Whether to auto-register built-in tools.
    max_retries:
        Maximum retry attempts (reserved for V2, not enforced in V1).

    """

    model_config = ConfigDict(frozen=True)

    default_timeout: float = 30.0
    default_permission_mode: str = "allow_all"
    builtin_tools_enabled: bool = True
    max_retries: int = Field(default=3, description="Reserved for V2 -- not enforced in V1.")
