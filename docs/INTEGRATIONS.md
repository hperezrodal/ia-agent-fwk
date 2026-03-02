# Channel Integrations

ia-agent-fwk provides pre-built channel integrations for Slack, Email, and WhatsApp. All channels share a unified interface and are routed through the `ChannelRouter`.

## Architecture

```
                    ┌──────────────────┐
   Slack webhook ──▶│                  │
   Email IMAP    ──▶│  Channel Router  │──▶ Agent ──▶ LLM
   WhatsApp hook ──▶│                  │
                    └──────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  Send Response   │
                    │  (via channel)   │
                    └──────────────────┘
```

The flow:
1. An incoming event (webhook payload, email, etc.) arrives
2. The `ChannelRouter` dispatches it to the appropriate `ChannelIntegration`
3. The integration normalizes the event into an `IncomingMessage`
4. The router creates an agent, executes it with the message content
5. The response is sent back through the same channel as an `OutgoingMessage`

## Channel Integration ABC

All channels implement the `ChannelIntegration` interface:

```python
class ChannelIntegration(ABC):
    @property
    @abstractmethod
    def channel_type(self) -> str: ...

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> bool: ...

    @abstractmethod
    async def process_incoming(self, raw_event: dict) -> IncomingMessage | None: ...

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def health_check(self) -> bool: ...
```

## Message Models

### IncomingMessage

Normalized inbound message from any channel:

| Field | Type | Description |
|---|---|---|
| `channel` | `str` | Channel type identifier |
| `sender` | `str` | Sender identifier |
| `content` | `str` | Message text content |
| `metadata` | `dict[str, str]` | Channel-specific metadata |
| `timestamp` | `str` | Message timestamp |
| `conversation_id` | `str \| None` | Conversation ID for threading |

### OutgoingMessage

Normalized outbound message to any channel:

| Field | Type | Description |
|---|---|---|
| `channel` | `str` | Channel type identifier |
| `recipient` | `str` | Recipient identifier |
| `content` | `str` | Message text content |
| `metadata` | `dict[str, str]` | Channel-specific metadata |
| `format` | `str` | Content format (default: `"text"`) |

## Slack Setup

### Prerequisites

- A Slack workspace
- A Slack app with Bot Token and Signing Secret
- Install the `slack` extra: `pip install ia-agent-fwk[slack]`

### Configuration

```yaml
integrations:
  slack:
    enabled: true
    bot_token: ""               # Set via IAFWK_INTEGRATIONS__SLACK__BOT_TOKEN
    signing_secret: ""          # Set via IAFWK_INTEGRATIONS__SLACK__SIGNING_SECRET
    default_agent: "customer_support"  # Agent type to handle Slack messages
```

```bash
export IAFWK_INTEGRATIONS__SLACK__BOT_TOKEN="xoxb-..."
export IAFWK_INTEGRATIONS__SLACK__SIGNING_SECRET="..."
```

### Slack App Setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Under **OAuth & Permissions**, add these Bot Token Scopes:
   - `chat:write` -- Send messages
   - `app_mentions:read` -- Read @mentions
   - `channels:history` -- Read channel messages
3. Install the app to your workspace and copy the Bot Token
4. Under **Basic Information**, copy the Signing Secret
5. Under **Event Subscriptions**, enable events and set the request URL to your API endpoint (e.g., `https://your-domain.com/api/v1/integrations/slack/events`)
6. Subscribe to bot events: `app_mention`, `message.channels`

### Using the Channel Router

```python
from ia_agent_fwk.integrations import ChannelRouter, ChannelConfig
from ia_agent_fwk.integrations.slack import SlackIntegration

router = ChannelRouter()

slack = SlackIntegration(
    bot_token="xoxb-...",
    signing_secret="...",
)

router.register(
    channel=slack,
    config=ChannelConfig(
        channel_type="slack",
        enabled=True,
        agent_type="customer_support",
    ),
)

# Route an incoming webhook event
response = await router.route_incoming(
    channel_type="slack",
    raw_event=webhook_payload,
    llm_settings=settings.llm,
)
```

## Email Setup

### Prerequisites

- An email account with SMTP and IMAP access
- Install the `email` extra: `pip install ia-agent-fwk[email]`

### Configuration

```yaml
integrations:
  email:
    enabled: true
    smtp:
      host: "smtp.gmail.com"
      port: 587
      username: ""              # Set via env var
      password: ""              # Set via env var
      use_tls: true
    imap:
      host: "imap.gmail.com"
      port: 993
      username: ""              # Set via env var
      password: ""              # Set via env var
      use_ssl: true
    default_agent: "customer_support"
```

```bash
export IAFWK_INTEGRATIONS__EMAIL__SMTP__USERNAME="agent@company.com"
export IAFWK_INTEGRATIONS__EMAIL__SMTP__PASSWORD="app-specific-password"
export IAFWK_INTEGRATIONS__EMAIL__IMAP__USERNAME="agent@company.com"
export IAFWK_INTEGRATIONS__EMAIL__IMAP__PASSWORD="app-specific-password"
```

### Gmail App Password

For Gmail, you need an App Password (not your regular password):
1. Enable 2-Step Verification on your Google Account
2. Go to **Security > App passwords**
3. Generate a new app password for "Mail"
4. Use this password in the configuration

## WhatsApp Setup

### Prerequisites

- A Meta Business account
- A WhatsApp Business API application
- Phone number registered with WhatsApp Business

### Configuration

```yaml
integrations:
  whatsapp:
    enabled: true
    api_url: "https://graph.facebook.com/v18.0"
    phone_number_id: ""         # Your WhatsApp phone number ID
    access_token: ""            # Set via IAFWK_INTEGRATIONS__WHATSAPP__ACCESS_TOKEN
    verify_token: ""            # Webhook verification token
    default_agent: "customer_support"
```

```bash
export IAFWK_INTEGRATIONS__WHATSAPP__ACCESS_TOKEN="EAA..."
export IAFWK_INTEGRATIONS__WHATSAPP__PHONE_NUMBER_ID="123456789"
export IAFWK_INTEGRATIONS__WHATSAPP__VERIFY_TOKEN="my-verify-token"
```

### WhatsApp Business API Setup

1. Create a Meta Business account at [business.facebook.com](https://business.facebook.com)
2. Create a WhatsApp Business app in the [Meta Developer Portal](https://developers.facebook.com)
3. Set up a phone number for the WhatsApp Business API
4. Generate a permanent access token
5. Configure the webhook URL (e.g., `https://your-domain.com/api/v1/integrations/whatsapp/webhook`)
6. Set the verify token to match your `verify_token` configuration
7. Subscribe to the `messages` webhook event

## Channel Router Configuration

The `ChannelRouter` manages all registered channels:

```python
from ia_agent_fwk.integrations import ChannelRouter, ChannelConfig

router = ChannelRouter()

# Register channels
router.register(slack_integration, ChannelConfig(
    channel_type="slack",
    enabled=True,
    agent_type="customer_support",
))

router.register(email_integration, ChannelConfig(
    channel_type="email",
    enabled=True,
    agent_type="customer_support",
))

# List registered channels
channels = router.list_channels()  # ["email", "slack"]

# Get a specific channel
slack = router.get_channel("slack")

# Route incoming events
response = await router.route_incoming(
    channel_type="slack",
    raw_event=payload,
    llm_settings=settings.llm,
)
```

The router:
1. Looks up the channel integration by type
2. Calls `process_incoming()` to normalize the event
3. Creates an agent of the configured `agent_type`
4. Runs the agent with the message content
5. Sends the response back via `send_message()`

## Creating a Custom Channel Integration

Implement the `ChannelIntegration` ABC:

```python
from ia_agent_fwk.integrations import ChannelIntegration, IncomingMessage, OutgoingMessage


class DiscordIntegration(ChannelIntegration):
    def __init__(self, bot_token: str) -> None:
        self._bot_token = bot_token

    @property
    def channel_type(self) -> str:
        return "discord"

    async def send_message(self, message: OutgoingMessage) -> bool:
        # Send message via Discord API
        # Return True on success
        return True

    async def process_incoming(self, raw_event: dict) -> IncomingMessage | None:
        # Parse Discord event payload
        # Return None if the event should be ignored (e.g., bot messages)
        return IncomingMessage(
            channel="discord",
            sender=raw_event.get("author", {}).get("id", ""),
            content=raw_event.get("content", ""),
        )

    async def start(self) -> None:
        # Initialize Discord connection
        pass

    async def stop(self) -> None:
        # Close Discord connection
        pass

    async def health_check(self) -> bool:
        # Check if Discord connection is alive
        return True
```

Register it with the router:

```python
router.register(
    DiscordIntegration(bot_token="..."),
    ChannelConfig(channel_type="discord", enabled=True, agent_type="my_agent"),
)
```

## API Endpoints

The framework exposes integration endpoints at `/api/v1/integrations/`:

- `POST /api/v1/integrations/{channel}/events` -- Receive webhook events
- `POST /api/v1/integrations/{channel}/webhook` -- Webhook verification and event handling

These endpoints are included in the FastAPI app via the `integrations` router.
