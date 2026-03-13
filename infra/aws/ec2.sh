#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# EC2 instance control — start, stop, status, ssh.
# Finds instance by project tag or EC2_INSTANCE_ID env var.
#
# Usage:
#   ./ec2.sh status     # Show instance state + IP
#   ./ec2.sh start      # Start the instance
#   ./ec2.sh stop       # Stop the instance (saves cost)
#   ./ec2.sh ssh        # SSH into the instance
#   ./ec2.sh ssh <cmd>  # Run a command via SSH
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env (AWS credentials)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Also load ansible .env (EC2_HOST, SSH_KEY_FILE)
if [ -f ../ansible/.env ]; then
    set -a
    source ../ansible/.env
    set +a
fi

# Check AWS credentials
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "ERROR: No AWS credentials found. Add to .env"
    exit 1
fi

# ── Helpers ─────────────────────────────────────────────────

get_key_name() {
    grep 'key_pair_name' terraform.tfvars 2>/dev/null \
        | sed 's/.*=.*"\(.*\)"/\1/' || echo "ia-agent-fwk"
}

# Extract valid EC2 instance ID (i- + hex). Strips junk so "1hi-0b1d52971075f225em1l" -> "i-0b1d52971075f225e"
sanitize_instance_id() {
    local raw
    raw=$(printf '%s' "$1" | tr -d '\r\n' | tr -d '[:cntrl:]')
    # EC2 instance IDs are i- followed by 8 or 17 hex chars; extract first match
    if [[ "$raw" =~ (i-[0-9a-f]{8,17}) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "$raw"
    fi
}

get_instance_id() {
    # Prefer explicit env var (sanitized)
    if [ -n "${EC2_INSTANCE_ID:-}" ]; then
        echo "$(sanitize_instance_id "$EC2_INSTANCE_ID")"
        return
    fi
    # Find by Name tag
    local id
    id=$(./awscli.sh ec2 describe-instances \
        --filters "Name=tag:Name,Values=ia-agent-fwk-llm-*" \
                  "Name=instance-state-name,Values=running,stopped,stopping,pending" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>&1 | grep -v '>>>' | tr -d '[:space:]')
    id=$(sanitize_instance_id "$id")
    if [ -z "$id" ] || [ "$id" = "None" ]; then
        echo "ERROR: No ia-agent-fwk instance found." >&2
        exit 1
    fi
    echo "$id"
}

get_public_ip() {
    if [ -n "${EC2_HOST:-}" ]; then
        echo "$EC2_HOST"
        return
    fi
    local ip
    ip=$(./awscli.sh ec2 describe-instances \
        --instance-ids "$(get_instance_id)" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>&1 | grep -v '>>>' | tr -d '[:space:]')
    echo "$ip"
}

# ── Commands ────────────────────────────────────────────────

cmd_status() {
    local instance_id
    instance_id=$(get_instance_id)
    echo "Instance: $instance_id"
    ./awscli.sh ec2 describe-instances \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].{State:State.Name,Type:InstanceType,LaunchTime:LaunchTime,PublicIp:PublicIpAddress,PrivateIp:PrivateIpAddress}' \
        --output table 2>&1 | grep -v '>>>'
}

cmd_start() {
    local instance_id
    instance_id=$(get_instance_id)
    echo "Starting $instance_id..."
    ./awscli.sh ec2 start-instances --instance-ids "$instance_id" 2>&1 | grep -v '>>>'
    echo ""
    echo "Waiting for instance to be running..."
    ./awscli.sh ec2 wait instance-running --instance-ids "$instance_id" 2>&1 | grep -v '>>>'
    echo "Instance is running."
    echo ""
    local ip key_name
    ip=$(get_public_ip)
    key_name=$(get_key_name)
    echo "Waiting for SSH ($ip)..."
    for i in $(seq 1 30); do
        if ssh -i "$HOME/.ssh/$key_name.pem" -o StrictHostKeyChecking=no -o ConnectTimeout=3 "ubuntu@$ip" "echo ready" 2>/dev/null; then
            echo ""
            echo "============================================"
            echo "  Instance ready!"
            echo "============================================"
            echo "  SSH:     ssh -i ~/.ssh/$key_name.pem ubuntu@$ip"
            echo "  API:     http://$ip:8000"
            echo "  Grafana: http://$ip:3000"
            echo "  Ollama:  http://$ip:11434"
            echo "============================================"
            return 0
        fi
        sleep 5
    done
    echo "WARNING: SSH not ready after 150s. Instance may still be booting."
}

cmd_stop() {
    local instance_id
    instance_id=$(get_instance_id)
    echo "Stopping $instance_id..."
    ./awscli.sh ec2 stop-instances --instance-ids "$instance_id" 2>&1 | grep -v '>>>'
    echo ""
    echo "Instance is stopping. Spot persistent — EIP preserved for restart."
}

cmd_ssh() {
    local ip key_name
    ip=$(get_public_ip)
    key_name=$(get_key_name)
    shift || true
    if [ $# -gt 0 ]; then
        ssh -i "$HOME/.ssh/$key_name.pem" -o StrictHostKeyChecking=no "ubuntu@$ip" "$@"
    else
        ssh -i "$HOME/.ssh/$key_name.pem" -o StrictHostKeyChecking=no "ubuntu@$ip"
    fi
}

# ── Main ────────────────────────────────────────────────────

case "${1:-}" in
    status) cmd_status ;;
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    ssh)    cmd_ssh "$@" ;;
    *)
        echo "Usage: $0 {status|start|stop|ssh [cmd]}"
        echo ""
        echo "  status  — Show instance state + IP"
        echo "  start   — Start instance + wait for SSH"
        echo "  stop    — Stop instance (saves \$\$)"
        echo "  ssh     — Open SSH session (or run cmd)"
        exit 1
        ;;
esac
