#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Terraform wrapper — runs Terraform inside Docker container.
# No local Terraform installation required.
#
# Usage:
#   ./tf.sh init          # Initialize Terraform
#   ./tf.sh plan          # Preview changes
#   ./tf.sh apply         # Apply changes
#   ./tf.sh destroy       # Tear down infrastructure
#   ./tf.sh output        # Show outputs (IPs, etc.)
#   ./tf.sh <any-cmd>     # Any Terraform command
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Load .env — local (infra/aws/.env) takes precedence
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check for terraform.tfvars
if [ ! -f terraform.tfvars ] && [ "$1" != "init" ]; then
    echo "ERROR: terraform.tfvars not found."
    echo "Copy the example and fill in your values:"
    echo "  cp terraform.tfvars.example terraform.tfvars"
    exit 1
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

# Build the docker run command
DOCKER_ARGS=(run --rm)
# Add -it only if running interactively
if [ -t 0 ]; then
    DOCKER_ARGS+=(-it)
fi
DOCKER_ARGS+=(
    -v "$SCRIPT_DIR:/workspace"
    -w /workspace
)

# Pass AWS env vars
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION AWS_PROFILE; do
    if [ -n "${!var:-}" ]; then
        DOCKER_ARGS+=(-e "$var")
    fi
done

DOCKER_ARGS+=(hashicorp/terraform:1.10)

echo ">>> terraform $*"
docker "${DOCKER_ARGS[@]}" "$@"
