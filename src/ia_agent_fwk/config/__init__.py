"""Configuration system for ia-agent-fwk.

Public API::

    from ia_agent_fwk.config import AppSettings, load_config

    settings = load_config()          # uses IAFWK_APP__ENVIRONMENT or "development"
    settings = load_config("testing") # explicit profile
"""

from ia_agent_fwk.config.loader import deep_merge, load_config
from ia_agent_fwk.config.settings import AppCoreSettings, AppSettings

__all__ = ["AppCoreSettings", "AppSettings", "deep_merge", "load_config"]
