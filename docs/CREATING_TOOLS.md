# Creating Tools

Tools extend an agent's capabilities beyond text generation. This guide covers how to create, register, and configure custom tools.

## Tool ABC Interface

Every tool extends the `Tool` abstract base class from `ia_agent_fwk.tools`:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from ia_agent_fwk.tools import Tool, ToolContext


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description (shown to the LLM)."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """Pydantic model class for input validation."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Pydantic model class for output validation."""
        ...

    @property
    def tags(self) -> list[str]:
        """Optional tags for categorization (default: empty list)."""
        return []

    @abstractmethod
    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        """Execute the tool with validated input."""
        ...
```

## Input and Output Schemas

Every tool defines its input and output as Pydantic models. These schemas serve two purposes:
1. Input validation before execution
2. OpenAI function-calling format generation for the LLM

```python
from pydantic import BaseModel, Field


class TranslateInput(BaseModel):
    """Input schema for the translation tool."""
    text: str = Field(description="The text to translate")
    target_language: str = Field(description="Target language code (e.g., 'es', 'fr', 'de')")
    source_language: str = Field(default="auto", description="Source language code or 'auto' for detection")


class TranslateOutput(BaseModel):
    """Output schema for the translation tool."""
    translated_text: str
    detected_language: str
    confidence: float
```

The `description` fields in the input schema are included in the OpenAI function-calling format, helping the LLM understand what each parameter expects.

## Minimal Tool Example

```python
from pydantic import BaseModel, Field
from ia_agent_fwk.tools import Tool, ToolContext


class ReverseInput(BaseModel):
    text: str = Field(description="The text to reverse")


class ReverseOutput(BaseModel):
    reversed_text: str


class ReverseTool(Tool):
    @property
    def name(self) -> str:
        return "reverse_text"

    @property
    def description(self) -> str:
        return "Reverse the characters in a text string."

    @property
    def input_schema(self) -> type[BaseModel]:
        return ReverseInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return ReverseOutput

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        inp = validated_input  # type: ReverseInput
        return ReverseOutput(reversed_text=inp.text[::-1])
```

## ToolContext

Every tool receives a `ToolContext` dataclass with execution metadata:

| Field | Type | Default | Description |
|---|---|---|---|
| `execution_id` | `str` | required | Unique ID for this execution |
| `agent_id` | `str` | `""` | ID of the agent executing the tool |
| `timeout` | `float` | `30.0` | Maximum execution time in seconds |
| `metadata` | `dict[str, Any]` | `{}` | Additional context data |

Use the context for logging, timeout awareness, or passing data between tools:

```python
async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
    logger.info("Tool '%s' called by agent '%s' (exec: %s)",
                self.name, context.agent_id, context.execution_id)
    # ... implementation
```

## Registration

### Register in a ToolRegistry

```python
from ia_agent_fwk.tools import ToolRegistry

registry = ToolRegistry()
registry.register(ReverseTool())

# Replace an existing tool
registry.register(ReverseTool(), replace=True)

# Check if a tool exists
assert registry.has("reverse_text")

# Get a tool by name
tool = registry.get("reverse_text")

# List all tools
all_tools = registry.list()

# List tools with specific tags
tagged_tools = registry.list(tags=["text", "utility"])

# Remove a tool
registry.remove("reverse_text")
```

### Register with Built-in Tools

To add your custom tool alongside the framework's built-in tools:

```python
from ia_agent_fwk.tools import ToolRegistry
from ia_agent_fwk.tools.builtin import register_builtin_tools

registry = ToolRegistry()
register_builtin_tools(registry)           # Adds calculator, http_request, etc.
registry.register(ReverseTool())           # Add your custom tool
```

## Permissions

The tool permission system controls which agents can use which tools.

### Permission Modes

| Mode | Description |
|---|---|
| `allow_all` | All tools are permitted (default) |
| `allow_list` | Only listed tools are permitted |
| `deny_list` | Listed tools are denied, all others permitted |
| `require_confirmation` | Listed tools require human confirmation |

### Configuring Permissions

```python
from ia_agent_fwk.tools import ToolPermissionManager
from ia_agent_fwk.tools.config import ToolPermissionConfig
from ia_agent_fwk.tools.permissions import PermissionMode

# Default: allow all
manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)

# Per-agent allow list
manager = ToolPermissionManager(
    default_mode=PermissionMode.deny_list,
    agent_permissions={
        "my-agent": ToolPermissionConfig(
            mode="allow_list",
            allowed=["calculator", "reverse_text"],
        ),
        "admin-agent": ToolPermissionConfig(
            mode="allow_all",
        ),
    },
)

# Check permissions programmatically
manager.check_permission("my-agent", "calculator")     # OK
manager.check_permission("my-agent", "http_request")   # raises ToolPermissionError

# Boolean check
is_ok = manager.is_permitted("my-agent", "calculator")  # True
```

### Via Configuration

```yaml
tools:
  default_permission_mode: "allow_list"
```

## Using Tags

Tags help categorize tools for filtering:

```python
class ReverseTool(Tool):
    @property
    def tags(self) -> list[str]:
        return ["text", "utility", "string-manipulation"]

    # ... rest of the implementation
```

Filter tools by tags:

```python
text_tools = registry.list(tags=["text"])
```

## OpenAI Function-Calling Schema

The `ToolRegistry` can export tools in OpenAI function-calling format:

```python
schemas = registry.openai_schemas()
# Returns:
# [
#     {
#         "type": "function",
#         "function": {
#             "name": "reverse_text",
#             "description": "Reverse the characters in a text string.",
#             "parameters": { ... JSON Schema from input_schema ... }
#         }
#     }
# ]

# Filter by agent permissions
schemas = registry.openai_schemas(
    agent_id="my-agent",
    permission_manager=permission_manager,
)
```

## Complete Example: API Lookup Tool

```python
import httpx
from pydantic import BaseModel, Field
from ia_agent_fwk.tools import Tool, ToolContext


class StockPriceInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL')")


class StockPriceOutput(BaseModel):
    symbol: str
    price: float
    currency: str
    source: str


class StockPriceTool(Tool):
    """Look up current stock prices."""

    @property
    def name(self) -> str:
        return "stock_price"

    @property
    def description(self) -> str:
        return "Get the current stock price for a given ticker symbol."

    @property
    def input_schema(self) -> type[BaseModel]:
        return StockPriceInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return StockPriceOutput

    @property
    def tags(self) -> list[str]:
        return ["finance", "api", "market-data"]

    async def execute(self, validated_input: BaseModel, context: ToolContext) -> BaseModel:
        inp: StockPriceInput = validated_input  # type: ignore[assignment]

        async with httpx.AsyncClient(timeout=context.timeout) as client:
            response = await client.get(
                f"https://api.example.com/stocks/{inp.symbol}",
            )
            data = response.json()

        return StockPriceOutput(
            symbol=inp.symbol.upper(),
            price=data["price"],
            currency=data.get("currency", "USD"),
            source="example-api",
        )
```

## Built-in Tools Reference

| Tool Name | Description |
|---|---|
| `calculator` | Evaluate mathematical expressions |
| `file_reader` | Read file contents from the filesystem |
| `http_request` | Make HTTP requests (GET, POST, etc.) |
| `web_scraper` | Extract content from web pages |
| `database_query` | Execute database queries |
| `current_time` | Get the current date and time |
| `echo` | Echo input back (useful for testing) |

The example agents add domain-specific tools:
- **Customer Support**: `ticket_lookup`, `faq_search`, `escalation`, `response_draft`
- **Document Processor**: document analysis tools
- **Finance**: financial data tools
