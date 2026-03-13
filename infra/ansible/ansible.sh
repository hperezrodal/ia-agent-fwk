#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Ansible wrapper — runs Ansible inside Docker container.
# No local Ansible installation required.
#
# Usage:
#   ./ansible.sh ansible all -m ping
#   ./ansible.sh ansible-playbook playbooks/deploy.yml
#   ./ansible.sh ansible-playbook playbooks/update.yml
#   ./ansible.sh ansible-playbook playbooks/teardown.yml
#   ./ansible.sh <any-ansible-command>
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check for required variables
if [ -z "${EC2_HOST:-}" ]; then
    echo "ERROR: EC2_HOST not set."
    echo "Add to your .env:"
    echo "  EC2_HOST=<your-ec2-ip>"
    exit 1
fi

if [ -z "${SSH_KEY_FILE:-}" ]; then
    echo "ERROR: SSH_KEY_FILE not set."
    echo "Add to your .env:"
    echo "  SSH_KEY_FILE=~/.ssh/your-key.pem"
    exit 1
fi

# Resolve SSH key path
SSH_KEY_RESOLVED="${SSH_KEY_FILE/#\~/$HOME}"
if [ ! -f "$SSH_KEY_RESOLVED" ]; then
    echo "ERROR: SSH key not found: $SSH_KEY_RESOLVED"
    exit 1
fi

# Build docker run command
DOCKER_ARGS=(run --rm)
# Add -it only if running interactively
if [ -t 0 ]; then
    DOCKER_ARGS+=(-it)
fi
DOCKER_ARGS+=(
    -v "$SCRIPT_DIR:/workspace"
    -v "$SSH_KEY_RESOLVED:/root/.ssh/id_rsa:ro"
    -w /workspace
    -e "EC2_HOST=${EC2_HOST}"
    -e "ANSIBLE_HOST_KEY_CHECKING=false"
)

# Pass optional env vars
for var in GITHUB_REPO GITHUB_BRANCH OLLAMA_MODEL OLLAMA_EMBED_MODEL IAFWK_API_KEYS GRAFANA_PASSWORD; do
    if [ -n "${!var:-}" ]; then
        DOCKER_ARGS+=(-e "$var")
    fi
done

DOCKER_ARGS+=(cytopia/ansible:latest-tools)

echo ">>> $*"
docker "${DOCKER_ARGS[@]}" "$@"
