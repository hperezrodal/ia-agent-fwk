#!/bin/bash
# ==============================================================================
# ia-agent-fwk -- Entrypoint script
# ==============================================================================
# Selects the service mode based on the first argument (or MODE env var).
# Modes: api, worker, beat
# ==============================================================================
set -e

MODE="${1:-${MODE:-api}}"

case "$MODE" in
  api)
    echo "Starting API server..."
    exec uvicorn ia_agent_fwk.api.app:create_app \
      --factory \
      --host "${API_HOST:-0.0.0.0}" \
      --port "${API_PORT:-8000}" \
      --workers "${UVICORN_WORKERS:-4}"
    ;;
  worker)
    echo "Starting Celery worker..."
    exec celery -A ia_agent_fwk.execution.celery_app worker \
      --loglevel="${LOG_LEVEL:-info}" \
      --concurrency="${CELERY_CONCURRENCY:-4}" \
      --pool=prefork
    ;;
  beat)
    echo "Starting Celery Beat scheduler..."
    exec celery -A ia_agent_fwk.execution.celery_app beat \
      --loglevel="${LOG_LEVEL:-info}"
    ;;
  *)
    echo "Unknown mode: $MODE (use: api, worker, beat)"
    exit 1
    ;;
esac
