#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AWS CLI wrapper — runs AWS CLI inside Docker container.
# No local AWS CLI installation required.
#
# Usage:
#   ./awscli.sh sts get-caller-identity
#   ./awscli.sh ec2 describe-instances
#   ./awscli.sh ec2 describe-key-pairs
#   ./awscli.sh <any-aws-command>
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env — local (infra/aws/.env) takes precedence
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check for AWS credentials
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "ERROR: No AWS credentials found."
    echo "Add to your project .env:"
    echo "  AWS_ACCESS_KEY_ID=..."
    echo "  AWS_SECRET_ACCESS_KEY=..."
    echo "  AWS_DEFAULT_REGION=us-east-1"
    exit 1
fi

# Build docker run command
DOCKER_ARGS=(run --rm)
# Add -it only if running interactively
if [ -t 0 ]; then
    DOCKER_ARGS+=(-it)
fi

# Pass AWS env vars (strip CR and other control chars to avoid "invalid XML character" in API requests)
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION AWS_PROFILE; do
    if [ -n "${!var:-}" ]; then
        val=$(printf '%s' "${!var}" | tr -d '\r' | tr -d '[:cntrl:]')
        DOCKER_ARGS+=(-e "$var=$val")
    fi
done

DOCKER_ARGS+=(amazon/aws-cli)

echo ">>> aws $*"
docker "${DOCKER_ARGS[@]}" "$@"
