#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Deploy script — build Docker image locally, push to VPS via SSH.
# No Docker registry needed.
#
# Usage:
#   ./docker/deploy.sh                    # Build + push + restart
#   ./docker/deploy.sh --build-only       # Just build locally
#   ./docker/deploy.sh --push-only        # Push existing image + restart
#   ./docker/deploy.sh --restart-only     # Just restart compose on VPS
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env from ansible dir (has EC2_HOST, SSH_KEY_FILE)
if [ -f infra/ansible/.env ]; then
    set -a
    source infra/ansible/.env
    set +a
fi

# ── Config ──────────────────────────────────────────────────
IMAGE_NAME="ia-agent-fwk"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
COMPOSE_FILE="docker/docker-compose.selfhosted-native-ollama.yml"
SSH_OPTS="-i ${SSH_KEY_FILE/#\~/$HOME} -o StrictHostKeyChecking=no"
SSH_TARGET="ubuntu@${EC2_HOST}"
REMOTE_DIR="/home/ubuntu/ia-agent-fwk"

# ── Validation ──────────────────────────────────────────────
if [ -z "${EC2_HOST:-}" ]; then
    echo "ERROR: EC2_HOST not set. Add to infra/ansible/.env"
    exit 1
fi
if [ -z "${SSH_KEY_FILE:-}" ]; then
    echo "ERROR: SSH_KEY_FILE not set. Add to infra/ansible/.env"
    exit 1
fi

# ── Parse args ──────────────────────────────────────────────
DO_BUILD=true
DO_PUSH=true
DO_RESTART=true

case "${1:-}" in
    --build-only)   DO_PUSH=false; DO_RESTART=false ;;
    --push-only)    DO_BUILD=false ;;
    --restart-only) DO_BUILD=false; DO_PUSH=false ;;
esac

# ── Build ───────────────────────────────────────────────────
if [ "$DO_BUILD" = true ]; then
    echo ">>> Building ${IMAGE}..."
    docker build -f docker/Dockerfile -t "$IMAGE" .
    echo ">>> Build complete"
fi

# ── Push via SSH pipe ───────────────────────────────────────
if [ "$DO_PUSH" = true ]; then
    echo ">>> Pushing ${IMAGE} to ${EC2_HOST}..."
    docker save "$IMAGE" | ssh $SSH_OPTS "$SSH_TARGET" "docker load"
    echo ">>> Push complete"
fi

# ── Sync compose + config files ─────────────────────────────
if [ "$DO_RESTART" = true ]; then
    echo ">>> Syncing config files..."
    rsync -az --rsync-path="mkdir -p ${REMOTE_DIR}/docker && rsync" \
        -e "ssh $SSH_OPTS" \
        docker/docker-compose.selfhosted-native-ollama.yml \
        "${SSH_TARGET}:${REMOTE_DIR}/docker/"

    # Sync observability configs
    for dir in prometheus grafana loki tempo promtail; do
        if [ -d "docker/$dir" ]; then
            rsync -az -e "ssh $SSH_OPTS" \
                "docker/$dir/" \
                "${SSH_TARGET}:${REMOTE_DIR}/docker/$dir/"
        fi
    done

    # ── Restart ─────────────────────────────────────────────
    echo ">>> Restarting stack on ${EC2_HOST}..."
    ssh $SSH_OPTS "$SSH_TARGET" bash -s <<REMOTE
        cd ${REMOTE_DIR}
        docker compose -f ${COMPOSE_FILE} up -d
        echo ">>> Waiting for API..."
        for i in \$(seq 1 30); do
            if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
                echo ">>> API healthy"
                break
            fi
            sleep 5
        done
        echo ""
        echo "============================================"
        echo "  Deployment complete!"
        echo "============================================"
        echo "  API:        http://${EC2_HOST}:8000"
        echo "  Grafana:    http://${EC2_HOST}:3000"
        echo "  Prometheus: http://${EC2_HOST}:9090"
        echo "  Ollama:     http://${EC2_HOST}:11434"
        echo "============================================"
REMOTE
fi
