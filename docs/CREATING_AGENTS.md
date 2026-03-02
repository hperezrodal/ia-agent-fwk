# Creating Agents

This guide covers how to build custom agents with ia-agent-fwk.

## Agent ABC Interface

Every agent extends the `Agent` abstract base class from `ia_agent_fwk.agents`. The only required override is the `agent_type` property:

```python
from ia_agent_fwk.agents import Agent

class MyAgent(Agent):
    @property
    def agent_type(self) -> str:
        return "my_agent"
```

The `Agent` class provides:
- **`run(input_text, conversation_history=None) -> AgentResult`** -- Executes the perceive-reason-act-observe loop
- **`pause()` / `resume(input_text)`** -- Pause and resume a running agent
- **`stop()`** -- Cancel a running agent
- **Lifecycle hooks** -- `on_start()`, `on_complete()`, `on_error()`, `on_timeout()`
- **State management** -- `IDLE -> RUNNING -> COMPLETED | FAILED | WAITING_FOR_INPUT`

## Minimal Agent Implementation

```python
import asyncio
from ia_agent_fwk.agents import Agent, AgentConfig, AgentRegistry
from ia_agent_fwk.config import load_config
from ia_agent_fwk.llm import LLMProviderFactory


class QAAgent(Agent):
    """A simple question-answering agent."""

    @property
    def agent_type(self) -> str:
        return "qa"


# Register the agent type so the factory can find it
AgentRegistry.register("qa", QAAgent)


async def main():
    settings = load_config()
    provider = LLMProviderFactory.create(settings.llm)

    config = AgentConfig(
        name="qa-agent",
        agent_type="qa",
        system_prompt="You are a knowledgeable assistant. Answer questions concisely.",
    )

    agent = QAAgent(config=config, provider=provider)
    result = await agent.run("What causes rainbows?")
    print(result.output)


asyncio.run(main())
```

## Agent Configuration

`AgentConfig` is a frozen Pydantic v2 model with these fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Unique agent instance name |
| `agent_type` | `str` | required | Agent type identifier (must match registry) |
| `system_prompt` | `str` | `""` | System prompt for the LLM |
| `provider_name` | `str` | `"openai"` | LLM provider to use |
| `model` | `str \| None` | `None` | Model override (uses provider default if None) |
| `max_iterations` | `int` | `10` | Maximum reasoning loop iterations |
| `execution_timeout` | `int` | `300` | Timeout in seconds |
| `max_tokens_per_response` | `int` | `4096` | Max tokens per LLM response |
| `tools` | `list[str]` | `[]` | Tool names this agent can use |
| `memory_config` | `dict \| None` | `None` | Memory backend configuration |
| `context_window` | `int \| None` | `None` | Context window size (defaults to 8192) |

## Lifecycle Hooks

Override these async methods to add custom behavior at different stages:

```python
from ia_agent_fwk.agents import Agent, AgentResult


class MonitoredAgent(Agent):
    @property
    def agent_type(self) -> str:
        return "monitored"

    async def on_start(self) -> None:
        """Called when the agent begins execution."""
        print(f"Agent '{self.config.name}' starting...")

    async def on_complete(self, result: AgentResult) -> None:
        """Called after successful completion."""
        print(f"Completed in {result.duration_ms:.0f}ms, {result.iterations} iterations")

    async def on_error(self, error: Exception) -> None:
        """Called when an error occurs (including timeout and max iterations)."""
        print(f"Error: {error}")

    async def on_timeout(self) -> None:
        """Called specifically on execution timeout (before on_error)."""
        print("Agent timed out!")
```

## Adding Tools

Tools give agents the ability to perform actions. There are two ways to equip an agent with tools.

### Using AgentFactory (Recommended)

The `AgentFactory` automatically creates a tool executor with built-in tools:

```python
from ia_agent_fwk.agents import AgentConfig, AgentFactory, AgentRegistry

AgentRegistry.register("my_agent", MyAgent)

config = AgentConfig(
    name="my-agent",
    agent_type="my_agent",
    tools=["calculator", "http_request", "current_time"],
)

agent = AgentFactory.create(config, settings.llm)
result = await agent.run("What is 42 * 17?")
```

Built-in tools: `calculator`, `file_reader`, `http_request`, `web_scraper`, `database_query`, `current_time`, `echo`.

### Manual Tool Setup

For more control, create the tool executor directly:

```python
from ia_agent_fwk.tools import ToolRegistry, DefaultToolExecutor, ToolPermissionManager
from ia_agent_fwk.tools.permissions import PermissionMode
from ia_agent_fwk.tools.builtin import register_builtin_tools

# Create registry with built-in tools
registry = ToolRegistry()
register_builtin_tools(registry)

# Add custom tools
registry.register(MyCustomTool())

# Create executor with permissions
permission_manager = ToolPermissionManager(default_mode=PermissionMode.allow_all)
executor = DefaultToolExecutor(
    registry=registry,
    permission_manager=permission_manager,
    agent_id="my-agent",
)

# Create agent with the executor
agent = MyAgent(config=config, provider=provider, tool_executor=executor)
```

## Using Memory

Agents can access memory backends for persistent storage.

### Conversation History

The REST API automatically manages conversation history using `ConversationMemoryBackend`. When using the agent programmatically, pass conversation history:

```python
from ia_agent_fwk.llm.models import Message

history = [
    Message(role="user", content="My name is Alice."),
    Message(role="assistant", content="Hello Alice! How can I help?"),
]

result = await agent.run(
    "What is my name?",
    conversation_history=history,
)
```

### Direct Memory Access

For custom memory operations, use a memory backend:

```python
from ia_agent_fwk.memory import MemoryFactory
from ia_agent_fwk.config import load_config

settings = load_config()
memory = MemoryFactory.create(settings.memory)

# Store and retrieve
await memory.store("user:alice:preferences", {"theme": "dark"})
prefs = await memory.retrieve("user:alice:preferences")

# Search (for vector backends)
results = await memory.search("user preferences", top_k=5)
```

## Pause and Resume

Agents support pausing mid-execution and resuming with new input:

```python
import asyncio

async def run_with_pause():
    agent = MyAgent(config=config, provider=provider)

    # Start in a background task
    task = asyncio.create_task(agent.run("Start processing..."))

    # Pause after some time
    await asyncio.sleep(2)
    agent.pause()  # State: RUNNING -> WAITING_FOR_INPUT

    # Resume with new input
    agent.resume("Continue with this additional context.")  # State: WAITING_FOR_INPUT -> RUNNING

    result = await task
    print(result.output)
```

## Stopping an Agent

```python
async def run_with_cancel():
    agent = MyAgent(config=config, provider=provider)
    task = asyncio.create_task(agent.run("Long running task..."))

    await asyncio.sleep(5)
    await agent.stop()  # Cancels the task, state -> FAILED

    result = await task
    print(result.error)  # "Agent stopped by user" or "Agent execution was cancelled"
```

## AgentResult

The `run()` method returns an `AgentResult` with these fields:

| Field | Type | Description |
|---|---|---|
| `output` | `str` | The agent's text output |
| `state` | `AgentState` | Final state (`COMPLETED` or `FAILED`) |
| `usage` | `TokenUsage` | Token usage (prompt, completion, total) |
| `iterations` | `int` | Number of reasoning loop iterations |
| `duration_ms` | `float` | Total execution time in milliseconds |
| `error` | `str \| None` | Error message if the agent failed |
| `metadata` | `dict \| None` | Optional additional metadata |

## Using the AgentFactory

The `AgentFactory.create()` method builds a fully configured agent from configuration:

```python
from ia_agent_fwk.agents import AgentConfig, AgentFactory

config = AgentConfig(
    name="my-agent",
    agent_type="qa",
    system_prompt="Be helpful.",
    provider_name="anthropic",
    max_iterations=5,
    execution_timeout=60,
)

# Creates the agent with LLM provider, tool executor, and permissions
agent = AgentFactory.create(
    config,
    settings.llm,
    tools_config=settings.tools,
    sandboxing_config=settings.security.tool_sandboxing,
)
```

## Complete Example: Customer Support Agent

See `examples/customer_support/agent.py` for a complete agent with:
- Custom system prompt
- Four domain-specific tools (ticket lookup, FAQ search, escalation, response drafting)
- Tool permission configuration
- Factory function for easy instantiation

```python
from examples.customer_support.agent import create_support_agent
from ia_agent_fwk.llm import LLMProviderFactory
from ia_agent_fwk.config import load_config

settings = load_config()
provider = LLMProviderFactory.create(settings.llm)
agent = create_support_agent(provider)

result = await agent.run("I need help with ticket TKT-001")
```
