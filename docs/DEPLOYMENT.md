# Deployment Guide

This guide covers running ia-agent-fwk in development and production environments.

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- An LLM provider API key (OpenAI, Anthropic) or a local Ollama instance

### Step 1: Start Infrastructure Services

```bash
docker compose up -d
```

This starts three services, all bound to `127.0.0.1` to prevent accidental network exposure:

| Service | Image | Port |
|---|---|---|
| PostgreSQL 16 + pgvector | `pgvector/pgvector:pg16` | `127.0.0.1:5434` |
| Redis 7 | `redis:7-alpine` | `127.0.0.1:6380` |
| Qdrant v1.12.6 | `qdrant/qdrant:v1.12.6` | `127.0.0.1:6333` |

Default PostgreSQL credentials: `postgres/postgres`, database: `ia_agent_fwk`.

### Step 2: Install the Framework

```bash
pip install -e ".[dev]"
```

### Step 3: Configure Environment

```bash
export IAFWK_LLM__PROVIDERS__OPENAI__API_KEY="sk-..."
```

### Step 4: Run the API Server

```bash
uvicorn ia_agent_fwk.api:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Step 5: Verify

```bash
curl http://localhost:8000/health
```

### Managing Infrastructure

```bash
# Stop services (preserve data)
docker compose down

# Stop and remove all data volumes
docker compose down -v

# View service logs
docker compose logs -f postgres
```

## Production Deployment

### Overview

The production stack (`docker/docker-compose.prod.yml`) runs all components as Docker containers:

- **API Server** -- FastAPI application serving REST endpoints
- **Celery Worker(s)** -- Background task processing
- **Celery Beat** -- Cron job scheduler
- **PostgreSQL 16 + pgvector** -- Primary database and vector store
- **Redis 7** -- Message broker and result backend (with persistence)
- **Qdrant** -- Vector database

Optional profiles:
- `--profile self-hosted` -- Adds Ollama for local LLM inference
- `--profile monitoring` -- Adds Flower for Celery monitoring UI

### Docker Image

The project uses a multi-stage Dockerfile (`docker/Dockerfile`):

```bash
# Build the image
docker build -f docker/Dockerfile -t ia-agent-fwk:latest .

# Run in different modes
docker run -e MODE=api    ia-agent-fwk:latest   # API server
docker run -e MODE=worker ia-agent-fwk:latest   # Celery worker
docker run -e MODE=beat   ia-agent-fwk:latest   # Celery beat
```

The image:
- Uses Python 3.11-slim base
- Runs as non-root user (`appuser`)
- Exposes port 8000
- Sets `PYTHONUNBUFFERED=1` for clean logging

### Required Environment Variables

Create a `.env` file or set these in your deployment environment:

```bash
# Database (required)
POSTGRES_PASSWORD=your-secure-password
POSTGRES_USER=postgres           # optional, defaults to postgres
POSTGRES_DB=ia_agent_fwk         # optional

# LLM Provider Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# API Authentication
IAFWK_API_KEYS=key1,key2,key3

# Scaling (optional)
UVICORN_WORKERS=4                # API server workers
WORKER_REPLICAS=1                # Number of Celery worker containers
CELERY_CONCURRENCY=4             # Tasks per worker
API_PORT=8000                    # External API port
LOG_LEVEL=info
```

### Deploy

```bash
# Start the full production stack
docker compose -f docker/docker-compose.prod.yml up -d

# With self-hosted Ollama
docker compose -f docker/docker-compose.prod.yml --profile self-hosted up -d

# With Celery monitoring (Flower UI on port 5555)
docker compose -f docker/docker-compose.prod.yml --profile monitoring up -d

# Scale workers
WORKER_REPLICAS=3 docker compose -f docker/docker-compose.prod.yml up -d
```

### Service Architecture

```
                    ┌──────────────────────────────┐
                    │        Load Balancer          │
                    │       (external, e.g.         │
                    │        Nginx / ALB)           │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     API Server (:8000)        │
                    │   uvicorn + FastAPI           │
                    │   N workers (UVICORN_WORKERS) │
                    └──────┬───────────┬───────────┘
                           │           │
              ┌────────────▼─┐   ┌─────▼──────────┐
              │   PostgreSQL │   │     Redis       │
              │  + pgvector  │   │  (broker +      │
              │              │   │   results)      │
              └──────────────┘   └──────┬──────────┘
                                        │
                           ┌────────────▼──────────┐
                           │   Celery Worker(s)     │
                           │   (background jobs)    │
                           └────────────────────────┘
                                        │
                           ┌────────────▼──────────┐
                           │   Celery Beat          │
                           │   (cron scheduler)     │
                           └────────────────────────┘
```

### Health Checks

All production services include Docker health checks:

- **API**: `curl -f http://localhost:8000/health`
- **PostgreSQL**: `pg_isready`
- **Redis**: `redis-cli ping`
- **Qdrant**: TCP check on port 6333

The API container depends on PostgreSQL and Redis being healthy before starting.

### Cloud Mode vs Self-Hosted Mode

**Cloud mode** (default): Uses cloud LLM APIs (OpenAI, Anthropic). Set the corresponding API keys.

**Self-hosted mode**: Uses Ollama for local LLM inference. Enable with:

```bash
docker compose -f docker/docker-compose.prod.yml --profile self-hosted up -d
```

Then configure the framework to use Ollama:

```bash
export IAFWK_LLM__DEFAULT_PROVIDER=ollama
```

Ollama data is persisted in a Docker volume. Pull models after startup:

```bash
docker compose -f docker/docker-compose.prod.yml exec ollama ollama pull llama3.1
```

### Production Configuration

Create a `config/production.yaml` to override defaults for production:

```yaml
app:
  environment: "production"
  debug: false
  log_level: "INFO"

server:
  workers: 8
  reload: false
  cors:
    allow_origins: ["https://your-domain.com"]

auth:
  enabled: true
  jwt:
    enabled: true
    algorithm: "HS256"
    expiration_minutes: 60
    # secret_key via IAFWK_AUTH__JWT__SECRET_KEY env var

observability:
  logging:
    format: "json"
    redact_pii: true
  tracing:
    enabled: true
    endpoint: "http://otel-collector:4317"
  metrics:
    enabled: true

security:
  rate_limiting:
    enabled: true
    default_rate: "100/minute"
  cost_controls:
    enabled: true
    max_tokens_per_day: 5000000
```

### Stopping Production

```bash
# Graceful shutdown (workers drain in-flight tasks for up to 300s)
docker compose -f docker/docker-compose.prod.yml down

# Remove all data (caution!)
docker compose -f docker/docker-compose.prod.yml down -v
```
