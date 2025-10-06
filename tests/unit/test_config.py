"""Tests for the configuration system: settings, loader, and profiles."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ia_agent_fwk.config import AppSettings, deep_merge, load_config
from ia_agent_fwk.config.profiles import (
    DEFAULT_ENVIRONMENT,
    VALID_ENVIRONMENTS,
    resolve_environment,
)

if TYPE_CHECKING:
    from pathlib import Path


# ==============================================================================
# AppSettings instantiation
# ==============================================================================


@pytest.mark.unit
class TestAppSettings:
    """Verify Pydantic Settings model defaults and env var support."""

    def test_default_settings_instantiation(self):
        """AppSettings() creates a valid instance with all defaults."""
        settings = AppSettings()
        assert settings.app.name == "ia-agent-fwk"
        assert settings.app.version == "0.1.0"
        assert settings.app.environment == "development"
        assert settings.app.debug is False
        assert settings.database.pool_size == 10
        assert settings.redis.url == "redis://localhost:6380/0"
        assert settings.memory.default_backend == "in_memory"

    def test_settings_has_env_prefix(self):
        """AppSettings uses IAFWK_ as env var prefix."""
        config = AppSettings.model_config
        assert config.get("env_prefix") == "IAFWK_"

    def test_settings_has_nested_delimiter(self):
        """AppSettings uses __ as the nested env var delimiter."""
        config = AppSettings.model_config
        assert config.get("env_nested_delimiter") == "__"

    def test_settings_frozen(self):
        """AppSettings is frozen; attribute assignment raises an error."""
        settings = AppSettings()
        with pytest.raises(ValidationError):
            settings.app = settings.app  # type: ignore[misc]

    def test_all_config_sections_present(self):
        """AppSettings has all expected top-level configuration sections."""
        settings = AppSettings()
        assert settings.app is not None
        assert settings.server is not None
        assert settings.auth is not None
        assert settings.database is not None
        assert settings.redis is not None
        assert settings.llm is not None
        assert settings.memory is not None
        assert settings.rag is not None
        assert settings.execution is not None
        assert settings.streaming is not None
        assert settings.plugins is not None
        assert settings.integrations is not None
        assert settings.observability is not None
        assert settings.security is not None

    def test_env_var_override_database_url(self, monkeypatch: pytest.MonkeyPatch):
        """IAFWK_DATABASE__URL env var overrides the database URL."""
        monkeypatch.setenv("IAFWK_DATABASE__URL", "postgresql+asyncpg://custom:pass@db:5432/mydb")
        settings = AppSettings()
        assert settings.database.url == "postgresql+asyncpg://custom:pass@db:5432/mydb"

    def test_env_var_override_app_debug(self, monkeypatch: pytest.MonkeyPatch):
        """IAFWK_APP__DEBUG env var overrides app.debug (F-002/F-003 fix verification)."""
        monkeypatch.setenv("IAFWK_APP__DEBUG", "true")
        settings = AppSettings()
        assert settings.app.debug is True

    def test_env_var_override_app_environment(self, monkeypatch: pytest.MonkeyPatch):
        """IAFWK_APP__ENVIRONMENT env var overrides app.environment."""
        monkeypatch.setenv("IAFWK_APP__ENVIRONMENT", "staging")
        settings = AppSettings()
        assert settings.app.environment == "staging"


# ==============================================================================
# Deep merge utility
# ==============================================================================


@pytest.mark.unit
class TestDeepMerge:
    """Verify the deep_merge function."""

    def test_deep_merge_nested_dicts(self):
        """Deep merge correctly handles nested dicts without losing sibling keys."""
        base = {
            "database": {"host": "localhost", "port": 5432, "name": "mydb"},
            "redis": {"url": "redis://localhost"},
        }
        override = {
            "database": {"port": 9999},
        }
        result = deep_merge(base, override)

        assert result["database"]["port"] == 9999
        assert result["database"]["host"] == "localhost"
        assert result["database"]["name"] == "mydb"
        assert result["redis"]["url"] == "redis://localhost"

    def test_deep_merge_override_leaf(self):
        """Deep merge replaces leaf values when overriding."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"a": 10, "b": {"c": 20}}
        result = deep_merge(base, override)

        assert result["a"] == 10
        assert result["b"]["c"] == 20
        assert result["b"]["d"] == 3

    def test_deep_merge_adds_new_keys(self):
        """Deep merge adds keys from override that don't exist in base."""
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"] == 2

    def test_deep_merge_does_not_mutate_inputs(self):
        """Deep merge does not modify the original dicts."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        base_copy = {"a": {"b": 1}}
        override_copy = {"a": {"c": 2}}

        deep_merge(base, override)

        assert base == base_copy
        assert override == override_copy


# ==============================================================================
# Config loader
# ==============================================================================


@pytest.mark.unit
class TestLoadConfig:
    """Verify YAML loading, profile merging, and env var overrides."""

    def test_load_config_default_yaml(self, tmp_config_dir: Path):
        """load_config() with default.yaml returns valid settings."""
        settings = load_config(environment="development", config_dir=tmp_config_dir)
        assert settings.app.name == "test-app"
        assert settings.database.url.startswith("postgresql+asyncpg://")

    def test_profile_merge_development(self, tmp_config_dir_with_profiles: Path):
        """Development profile overrides app.debug to True."""
        settings = load_config(
            environment="development",
            config_dir=tmp_config_dir_with_profiles,
        )
        assert settings.app.debug is True
        assert settings.server.reload is True
        assert settings.app.log_level == "DEBUG"

    def test_profile_merge_testing(self, tmp_config_dir_with_profiles: Path):
        """Testing profile sets memory.default_backend to in_memory."""
        settings = load_config(
            environment="testing",
            config_dir=tmp_config_dir_with_profiles,
        )
        assert settings.app.environment == "testing"
        assert settings.memory.default_backend == "in_memory"

    def test_env_var_override(self, tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Env vars override YAML values."""
        monkeypatch.setenv("IAFWK_DATABASE__URL", "postgresql+asyncpg://envuser:pass@envhost:5555/envdb")
        settings = load_config(environment="development", config_dir=tmp_config_dir)
        assert settings.database.url == "postgresql+asyncpg://envuser:pass@envhost:5555/envdb"

    def test_env_var_precedence_over_yaml(
        self,
        tmp_config_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """IAFWK_APP__DEBUG env var takes precedence over YAML value (F-013 fix)."""
        monkeypatch.setenv("IAFWK_APP__DEBUG", "true")
        settings = load_config(environment="development", config_dir=tmp_config_dir)
        assert settings.app.debug is True

    def test_invalid_config_raises_validation_error(self, tmp_path: Path):
        """Providing an invalid type raises ValidationError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        default_yaml = config_dir / "default.yaml"
        default_yaml.write_text(
            textwrap.dedent("""\
                app:
                  name: "test"
                  debug: false

                database:
                  pool_size: "not_a_number"
            """),
        )
        with pytest.raises(ValidationError):
            load_config(environment="development", config_dir=config_dir)

    def test_missing_default_yaml_uses_pydantic_defaults(self, tmp_path: Path):
        """When default.yaml does not exist, settings fall back to Pydantic defaults."""
        config_dir = tmp_path / "empty_config"
        config_dir.mkdir()
        settings = load_config(environment="development", config_dir=config_dir)
        assert settings.app.name == "ia-agent-fwk"
        assert settings.app.version == "0.1.0"

    def test_missing_profile_yaml_uses_defaults_only(self, tmp_config_dir: Path):
        """When profile YAML does not exist, only default values are used."""
        settings = load_config(environment="staging", config_dir=tmp_config_dir)
        assert settings.app.name == "test-app"
        assert settings.app.debug is False

    def test_load_config_with_real_config_dir(self):
        """load_config() works with the real config/ directory."""
        settings = load_config(environment="development", config_dir="config")
        assert settings.app.name == "ia-agent-fwk"
        assert settings.app.debug is True


# ==============================================================================
# Semantic validation (F-007)
# ==============================================================================


@pytest.mark.unit
class TestSemanticValidation:
    """Verify semantic validation for production constraints."""

    def test_production_debug_true_raises(self, tmp_path: Path):
        """Production profile with debug=true raises ValueError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            textwrap.dedent("""\
                app:
                  environment: "production"
                  debug: true
                auth:
                  enabled: true
            """),
        )
        with pytest.raises(ValueError, match=r"app\.debug must be False in production"):
            load_config(environment="production", config_dir=config_dir)

    def test_production_auth_disabled_raises(self, tmp_path: Path):
        """Production profile with auth disabled raises ValueError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            textwrap.dedent("""\
                app:
                  environment: "production"
                  debug: false
                auth:
                  enabled: false
            """),
        )
        with pytest.raises(ValueError, match=r"auth\.enabled must be True in production"):
            load_config(environment="production", config_dir=config_dir)

    def test_production_jwt_enabled_without_secret_raises(self, tmp_path: Path):
        """Production with JWT enabled but no secret raises ValueError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            textwrap.dedent("""\
                app:
                  environment: "production"
                  debug: false
                auth:
                  enabled: true
                  jwt:
                    enabled: true
                    secret_key: ""
            """),
        )
        with pytest.raises(ValueError, match=r"auth\.jwt\.secret_key must be set"):
            load_config(environment="production", config_dir=config_dir)

    def test_non_production_allows_debug(self, tmp_path: Path):
        """Non-production environments allow debug mode without error."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            textwrap.dedent("""\
                app:
                  environment: "development"
                  debug: true
            """),
        )
        settings = load_config(environment="development", config_dir=config_dir)
        assert settings.app.debug is True


# ==============================================================================
# Environment profile resolution (F-019)
# ==============================================================================


@pytest.mark.unit
class TestResolveEnvironment:
    """Verify resolve_environment() logic."""

    def test_default_environment_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        """Returns 'development' when IAFWK_APP__ENVIRONMENT is not set."""
        monkeypatch.delenv("IAFWK_APP__ENVIRONMENT", raising=False)
        assert resolve_environment() == DEFAULT_ENVIRONMENT

    def test_default_environment_when_empty(self, monkeypatch: pytest.MonkeyPatch):
        """Returns 'development' when IAFWK_APP__ENVIRONMENT is empty."""
        monkeypatch.setenv("IAFWK_APP__ENVIRONMENT", "")
        assert resolve_environment() == DEFAULT_ENVIRONMENT

    @pytest.mark.parametrize("env", sorted(VALID_ENVIRONMENTS))
    def test_valid_environments(self, env: str, monkeypatch: pytest.MonkeyPatch):
        """Valid environment names are returned correctly."""
        monkeypatch.setenv("IAFWK_APP__ENVIRONMENT", env)
        assert resolve_environment() == env

    def test_invalid_environment_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Invalid environment name raises ValueError."""
        monkeypatch.setenv("IAFWK_APP__ENVIRONMENT", "invalid_env")
        with pytest.raises(ValueError, match="Invalid environment 'invalid_env'"):
            resolve_environment()


# ==============================================================================
# Deeply nested env var override (F-004 / F-009)
# ==============================================================================


@pytest.mark.unit
class TestDeeplyNestedEnvVar:
    """Verify env vars for dict-typed nested fields like llm.providers."""

    def test_deeply_nested_env_var_override(self, monkeypatch: pytest.MonkeyPatch):
        """IAFWK_LLM__PROVIDERS__OPENAI__API_KEY sets the provider key."""
        monkeypatch.setenv("IAFWK_LLM__PROVIDERS__OPENAI__API_KEY", "sk-test-key-123")
        settings = AppSettings()
        assert settings.llm.providers["openai"].api_key.get_secret_value() == "sk-test-key-123"
