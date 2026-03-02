# Creating Plugins

Plugins extend ia-agent-fwk with custom tools that can be packaged, distributed, and loaded dynamically. This guide covers the plugin system from creation to distribution.

## Plugin ABC Interface

Every plugin extends the `Plugin` abstract base class:

```python
from ia_agent_fwk.plugins import Plugin, PluginManifest
from ia_agent_fwk.tools import Tool


class Plugin(ABC):
    @property
    @abstractmethod
    def manifest(self) -> PluginManifest:
        """Return the plugin manifest describing identity and capabilities."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return tool instances to register in the framework."""
        ...

    async def on_load(self) -> None:
        """Lifecycle hook called after the plugin is loaded (optional)."""

    async def on_unload(self) -> None:
        """Lifecycle hook called before the plugin is unloaded (optional)."""
```

## Plugin Manifest

The `PluginManifest` is a frozen Pydantic model that describes a plugin:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique plugin identifier |
| `version` | `str` | `"0.1.0"` | Semantic version |
| `description` | `str` | `""` | Human-readable description |
| `author` | `str` | `""` | Plugin author |
| `tools` | `list[str]` | `[]` | Tool names the plugin provides |
| `entry_point` | `str` | `""` | Dotted path to the plugin class |
| `dependencies` | `list[str]` | `[]` | Required Python packages |

## Creating a Plugin

### Step 1: Define Your Tools

```python
# my_plugin/tools.py
from pydantic import BaseModel, Field
from ia_agent_fwk.tools import Tool, ToolContext


class SentimentInput(BaseModel):
    text: str = Field(description="Text to analyze for sentiment")


class SentimentOutput(BaseModel):
    sentiment: str
    score: float
    confidence: float


class SentimentTool(Tool):
    @property
    def name(self) -> str:
        return "sentiment_analysis"

    @property
    def description(self) -> str:
        return "Analyze the sentiment of a text (positive, negative, neutral)."

    @property
    def input_schema(self) -> type[BaseModel]:
        return SentimentInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return SentimentOutput

    @property
    def tags(self) -> list[str]:
        return ["nlp", "analysis"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        inp: SentimentInput = validated_input  # type: ignore[assignment]
        # Your sentiment analysis logic here
        return SentimentOutput(sentiment="positive", score=0.85, confidence=0.92)
```

### Step 2: Create the Plugin Class

```python
# my_plugin/plugin.py
from ia_agent_fwk.plugins import Plugin, PluginManifest
from ia_agent_fwk.tools import Tool

from my_plugin.tools import SentimentTool


class SentimentPlugin(Plugin):
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name="sentiment-plugin",
            version="1.0.0",
            description="Adds sentiment analysis capabilities to agents.",
            author="Your Name",
            tools=["sentiment_analysis"],
            entry_point="my_plugin.plugin:SentimentPlugin",
            dependencies=["textblob>=0.17"],
        )

    def get_tools(self) -> list[Tool]:
        return [SentimentTool()]

    async def on_load(self) -> None:
        """Initialize NLP models or connections."""
        print("Sentiment plugin loaded")

    async def on_unload(self) -> None:
        """Clean up resources."""
        print("Sentiment plugin unloaded")
```

### Step 3: Package the Plugin

Create a standard Python package with `pyproject.toml`:

```
my-sentiment-plugin/
  pyproject.toml
  my_plugin/
    __init__.py
    plugin.py
    tools.py
```

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ia-agent-fwk-sentiment"
version = "1.0.0"
dependencies = [
    "ia-agent-fwk>=0.1.0",
    "textblob>=0.17",
]

[project.entry-points."ia_agent_fwk.plugins"]
sentiment = "my_plugin.plugin:SentimentPlugin"
```

The entry point group must be `"ia_agent_fwk.plugins"`.

## Discovery and Loading

### Entry Points Discovery (Default)

When `discovery_method: "entry_points"` is configured (the default), the framework discovers plugins via Python entry points. Any installed package with an `ia_agent_fwk.plugins` entry point will be discovered automatically.

```bash
# Install the plugin package
pip install ia-agent-fwk-sentiment
```

### Directory Discovery

When `discovery_method: "directory"` is configured, the framework scans directories for plugin modules:

```yaml
plugins:
  discovery_method: "directory"
  plugin_dir: "./plugins"
  plugin_dirs:
    - "/opt/plugins"
    - "./custom_plugins"
```

Place your plugin module in the configured directory.

### Programmatic Discovery

```python
from ia_agent_fwk.plugins import (
    discover_plugins_from_entry_points,
    discover_plugins_from_directory,
)

# Discover from entry points
plugins = discover_plugins_from_entry_points()

# Discover from a directory
plugins = discover_plugins_from_directory("./plugins")
```

### Using the PluginLoader

```python
from ia_agent_fwk.plugins import PluginLoader

loader = PluginLoader()

# Load a plugin from a dotted path
plugin = loader.load("my_plugin.plugin:SentimentPlugin")
```

### Using the PluginManager

The `PluginManager` handles the full lifecycle of plugins:

```python
from ia_agent_fwk.plugins import PluginManager, PluginConfig

manager = PluginManager()

# Load a plugin
await manager.load_plugin("my_plugin.plugin:SentimentPlugin")

# Get plugin info
info = manager.get_plugin_info("sentiment-plugin")
print(info.tools_registered)  # ["sentiment-plugin:sentiment_analysis"]

# List all plugins
all_info = manager.list_plugins()

# Unload a plugin
await manager.unload_plugin("sentiment-plugin")
```

## Plugin Configuration

Plugins can receive configuration from the main YAML config:

```yaml
plugins:
  enabled: true
  plugins:
    - name: "sentiment-plugin"
      enabled: true
      settings:
        model: "large"
        language: "en"
        threshold: 0.5
```

The `PluginConfig` model provides:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Must match the plugin's manifest name |
| `enabled` | `bool` | `true` | Whether to load this plugin |
| `settings` | `dict[str, Any]` | `{}` | Arbitrary key-value settings |

## Tool Namespacing

When a plugin's tools are registered through the `PluginManager`, they are namespaced with the plugin name to avoid collisions:

- Plugin tool: `sentiment_analysis`
- Registered as: `sentiment-plugin:sentiment_analysis`

Agents reference the namespaced name:

```yaml
agents:
  agents:
    my_agent:
      tools:
        - "sentiment-plugin:sentiment_analysis"
        - "calculator"  # Built-in tools have no namespace prefix
```

## PluginInfo

The `PluginInfo` model provides a read-only snapshot of a loaded plugin:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Plugin name |
| `version` | `str` | Plugin version |
| `description` | `str` | Plugin description |
| `enabled` | `bool` | Whether the plugin is enabled |
| `tools_registered` | `list[str]` | Namespaced tool names registered |
| `load_status` | `str` | `loaded`, `unloaded`, or `error` |

## Error Handling

The plugin system defines four exception types:

| Exception | When |
|---|---|
| `PluginError` | Base exception for all plugin errors |
| `PluginLoadError` | Plugin class cannot be imported or instantiated |
| `PluginNotFoundError` | Plugin name not found in the manager |
| `PluginConfigError` | Invalid plugin configuration |

```python
from ia_agent_fwk.plugins import PluginLoadError

try:
    await manager.load_plugin("nonexistent.module:Plugin")
except PluginLoadError as e:
    print(f"Failed to load plugin: {e}")
```

## Complete Example

```python
"""Weather plugin providing a weather lookup tool."""
from pydantic import BaseModel, Field
from ia_agent_fwk.plugins import Plugin, PluginManifest
from ia_agent_fwk.tools import Tool, ToolContext


class WeatherInput(BaseModel):
    city: str = Field(description="City name")
    units: str = Field(default="celsius", description="Temperature units: celsius or fahrenheit")


class WeatherOutput(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int


class WeatherTool(Tool):
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather for a city."

    @property
    def input_schema(self) -> type[BaseModel]:
        return WeatherInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return WeatherOutput

    @property
    def tags(self) -> list[str]:
        return ["weather", "api"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        inp: WeatherInput = validated_input  # type: ignore[assignment]
        # Call weather API...
        return WeatherOutput(
            city=inp.city, temperature=22.0, condition="Sunny", humidity=45,
        )


class WeatherPlugin(Plugin):
    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            name="weather-plugin",
            version="1.0.0",
            description="Weather lookup tool.",
            author="Your Name",
            tools=["weather"],
            entry_point="weather_plugin:WeatherPlugin",
        )

    def get_tools(self) -> list[Tool]:
        return [WeatherTool()]
```
