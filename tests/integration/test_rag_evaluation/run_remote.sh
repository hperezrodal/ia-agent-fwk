#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Run RAG evaluation tests against a remote EC2 instance.
# Opens SSH tunnels for Qdrant, runs pytest, cleans up.
#
# Usage:
#   ./tests/integration/test_rag_evaluation/run_remote.sh
#   ./tests/integration/test_rag_evaluation/run_remote.sh --hallucination
#   ./tests/integration/test_rag_evaluation/run_remote.sh --all
#
# Environment (defaults from infra/ansible/.env):
#   EC2_HOST         Remote IP
#   SSH_KEY_FILE     SSH key path
#   RAG_EVAL_MAX_DOCS    Number of docs to sample (default: 5)
#   RAG_EVAL_LLM_MODEL  LLM model for judge (default: qwen3:14b)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

# Load ansible .env for EC2_HOST and SSH_KEY_FILE
if [ -f infra/ansible/.env ]; then
    set -a
    source infra/ansible/.env
    set +a
fi

# ── Config ──────────────────────────────────────────────────
EC2_HOST="${EC2_HOST:?EC2_HOST not set}"
SSH_KEY="${SSH_KEY_FILE/#\~/$HOME}"
LOCAL_QDRANT_PORT=16333
MAX_DOCS="${RAG_EVAL_MAX_DOCS:-5}"
LLM_MODEL="${RAG_EVAL_LLM_MODEL:-qwen3:14b}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"

# ── Select tests ────────────────────────────────────────────
TESTS="test_rag_bench_ollama"
case "${1:-}" in
    --hallucination) TESTS="test_rag_bench_hallucination" ;;
    --all)           TESTS="test_rag_bench_ollama test_rag_bench_hallucination" ;;
    --chunking)      TESTS="test_chunking_comparison" ;;
esac

TEST_ARGS=""
for t in $TESTS; do
    TEST_ARGS="$TEST_ARGS tests/integration/test_rag_evaluation/test_rag_bench.py::$t"
done

# ── Preflight checks ───────────────────────────────────────
echo ">>> Checking remote services on $EC2_HOST..."

if ! ssh $SSH_OPTS "ubuntu@$EC2_HOST" "curl -sf http://localhost:11434/api/tags >/dev/null" 2>/dev/null; then
    echo "ERROR: Ollama not responding on $EC2_HOST:11434"
    exit 1
fi

if ! ssh $SSH_OPTS "ubuntu@$EC2_HOST" "curl -sf http://localhost:6333/healthz >/dev/null" 2>/dev/null; then
    echo "ERROR: Qdrant not responding on $EC2_HOST (container port 6333)"
    exit 1
fi

echo "    Ollama: OK"
echo "    Qdrant: OK"

# ── Open SSH tunnel ─────────────────────────────────────────
# Kill any existing tunnel on the local port
kill "$(lsof -ti:$LOCAL_QDRANT_PORT)" 2>/dev/null || true
sleep 1

echo ">>> Opening SSH tunnel (localhost:$LOCAL_QDRANT_PORT -> $EC2_HOST:6333)..."
ssh $SSH_OPTS -f -N -L "$LOCAL_QDRANT_PORT:localhost:6333" "ubuntu@$EC2_HOST"
TUNNEL_PID=$(lsof -ti:$LOCAL_QDRANT_PORT 2>/dev/null || true)

# Ensure tunnel is cleaned up on exit
cleanup() {
    echo ""
    echo ">>> Closing SSH tunnel (pid: $TUNNEL_PID)..."
    kill "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Verify tunnel
sleep 1
if ! curl -sf "http://127.0.0.1:$LOCAL_QDRANT_PORT/healthz" >/dev/null; then
    echo "ERROR: SSH tunnel not working"
    exit 1
fi
echo "    Tunnel: OK"

# ── Run tests ───────────────────────────────────────────────
echo ""
echo ">>> Running: $TESTS"
echo "    Ollama:  http://$EC2_HOST:11434"
echo "    Qdrant:  http://127.0.0.1:$LOCAL_QDRANT_PORT (tunneled)"
echo "    Model:   $LLM_MODEL"
echo "    Docs:    $MAX_DOCS"
echo ""

RAG_EVAL_OLLAMA_URL="http://$EC2_HOST:11434" \
RAG_EVAL_QDRANT_URL="http://127.0.0.1:$LOCAL_QDRANT_PORT" \
RAG_EVAL_MAX_DOCS="$MAX_DOCS" \
RAG_EVAL_LLM_MODEL="$LLM_MODEL" \
python3 -m pytest $TEST_ARGS -v -s --log-cli-level=INFO
