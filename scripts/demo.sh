#!/bin/bash
# ==============================================================================
# ia-agent-fwk -- Self-Hosted Demo Script
# ==============================================================================
# Demonstrates the 3 example agents running fully self-hosted with Ollama.
# No external API calls. All AI inference runs locally.
#
# Usage:
#   ./scripts/demo.sh setup     # Start infra + pull model (first time)
#   ./scripts/demo.sh start     # Start infra (model already pulled)
#   ./scripts/demo.sh demo      # Run all 3 agent demos
#   ./scripts/demo.sh chat      # Interactive chat with an agent
#   ./scripts/demo.sh stop      # Stop everything
#   ./scripts/demo.sh status    # Check service status
# ==============================================================================
set -e

COMPOSE_FILE="docker/docker-compose.selfhosted.yml"
API_URL="http://localhost:8000"
API_KEY="${IAFWK_API_KEYS:-dev-api-key}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "\n${GREEN}▶ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}  ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

wait_for_api() {
    print_step "Waiting for API to be ready..."
    for i in $(seq 1 60); do
        if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
            print_info "API is ready!"
            return 0
        fi
        sleep 2
    done
    print_error "API did not become ready in time"
    return 1
}

wait_for_ollama() {
    print_step "Waiting for Ollama to be ready..."
    for i in $(seq 1 60); do
        if docker compose -f "$COMPOSE_FILE" exec -T ollama curl -sf http://localhost:11434/ > /dev/null 2>&1; then
            print_info "Ollama is ready!"
            return 0
        fi
        sleep 2
    done
    print_error "Ollama did not become ready in time"
    return 1
}

call_agent() {
    local agent_type="$1"
    local prompt="$2"
    local conv_id="$3"

    local payload
    if [ -n "$conv_id" ]; then
        payload="{\"prompt\": \"$prompt\", \"conversation_id\": \"$conv_id\"}"
    else
        payload="{\"prompt\": \"$prompt\"}"
    fi

    curl -s -X POST "${API_URL}/api/v1/agents/${agent_type}/run" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${API_KEY}" \
        -d "$payload"
}

pretty_response() {
    local response="$1"
    local output
    output=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'error' in data:
        print(f\"ERROR: {data['error'].get('message', data['error'])}\")
    else:
        print(data.get('output', 'No output'))
        print(f\"\\n  [iterations: {data.get('iterations', '?')}, duration: {data.get('duration_ms', '?'):.0f}ms]\")
except Exception as e:
    print(f'Parse error: {e}')
" 2>&1)
    echo "$output"
}

get_conversation_id() {
    local response="$1"
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('conversation_id', ''))
except:
    print('')
" 2>&1
}

# ==============================================================================
# Commands
# ==============================================================================

cmd_setup() {
    print_header "ia-agent-fwk Self-Hosted Setup"

    print_step "Building and starting services..."
    docker compose -f "$COMPOSE_FILE" up -d --build

    wait_for_ollama

    print_step "Pulling Ollama model: ${OLLAMA_MODEL}..."
    print_info "This may take a few minutes on the first run."
    docker compose -f "$COMPOSE_FILE" exec ollama ollama pull "$OLLAMA_MODEL"

    wait_for_api

    print_step "Setup complete!"
    print_info "API available at: ${API_URL}"
    print_info "API Key: ${API_KEY}"
    print_info "Model: ${OLLAMA_MODEL}"
    echo ""
    echo -e "Run ${BOLD}./scripts/demo.sh demo${NC} to see the agents in action."
}

cmd_start() {
    print_header "Starting ia-agent-fwk"
    docker compose -f "$COMPOSE_FILE" up -d
    wait_for_api
    print_step "Ready!"
    print_info "API available at: ${API_URL}"
}

cmd_stop() {
    print_header "Stopping ia-agent-fwk"
    docker compose -f "$COMPOSE_FILE" down
    print_step "All services stopped."
}

cmd_status() {
    print_header "Service Status"
    docker compose -f "$COMPOSE_FILE" ps

    echo ""
    print_step "Health check:"
    curl -s "${API_URL}/health" | python3 -m json.tool 2>/dev/null || print_error "API is not responding"

    echo ""
    print_step "Ollama models:"
    docker compose -f "$COMPOSE_FILE" exec -T ollama ollama list 2>/dev/null || print_error "Ollama is not responding"
}

cmd_demo() {
    print_header "ia-agent-fwk — Agent Demo"
    echo -e "  Running 3 example agents, fully self-hosted with Ollama."
    echo -e "  No external API calls. ${BOLD}Complete AI privacy.${NC}"

    # ---- Customer Support Agent ----
    print_header "1/3  Customer Support Agent"
    print_info "This agent handles customer inquiries using ticket lookup, FAQ search, escalation, and response drafting tools."

    echo -e "\n${BOLD}Customer:${NC} I'm having trouble logging in to my account. Can you help?"
    print_step "Agent is thinking..."
    response=$(call_agent "customer_support" "I'm having trouble logging in to my account. Can you help?")
    echo -e "\n${BOLD}Agent:${NC}"
    pretty_response "$response"

    conv_id=$(get_conversation_id "$response")

    echo -e "\n${BOLD}Customer:${NC} My ticket ID is TKT-001, can you check the status?"
    print_step "Agent is thinking..."
    response=$(call_agent "customer_support" "My ticket ID is TKT-001, can you check the status?" "$conv_id")
    echo -e "\n${BOLD}Agent:${NC}"
    pretty_response "$response"

    # ---- Document Processor Agent ----
    print_header "2/3  Document Processor Agent"
    print_info "This agent extracts and analyzes information from documents."

    local doc_text="CONSULTING SERVICES AGREEMENT\n\nThis Agreement is entered into as of January 15, 2024, by and between TechCorp Inc. (the Client) and DataPros LLC (the Consultant).\n\nThe total project fee shall be \$150,000.00, payable in three installments.\n\nContact: john.doe@techcorp.com"

    echo -e "\n${BOLD}User:${NC} Analyze this contract and extract key information: parties, dates, amounts, and contact details."
    print_step "Agent is thinking..."
    response=$(call_agent "document_processor" "Analyze this contract and extract key information (parties, dates, amounts, contact emails): ${doc_text}")
    echo -e "\n${BOLD}Agent:${NC}"
    pretty_response "$response"

    # ---- Finance Agent ----
    print_header "3/3  Finance Agent"
    print_info "This agent analyzes financial data and detects anomalies."

    echo -e "\n${BOLD}User:${NC} Calculate the profit margin if revenue is 500000 and net income is 75000. Is this healthy?"
    print_step "Agent is thinking..."
    response=$(call_agent "finance" "Calculate the profit margin if revenue is 500000 and net income is 75000. Is this a healthy margin for a mid-market company?")
    echo -e "\n${BOLD}Agent:${NC}"
    pretty_response "$response"

    print_header "Demo Complete"
    echo -e "  All 3 agents ran ${BOLD}100% self-hosted${NC} using Ollama (${OLLAMA_MODEL})."
    echo -e "  No data left your network. ${BOLD}Complete AI privacy.${NC}"
    echo ""
    echo -e "  Run ${BOLD}./scripts/demo.sh chat${NC} for an interactive session."
}

cmd_chat() {
    print_header "Interactive Agent Chat"
    echo -e "  Available agents: ${BOLD}customer_support${NC}, ${BOLD}document_processor${NC}, ${BOLD}finance${NC}"
    echo ""

    read -p "Select agent [customer_support]: " agent_type
    agent_type="${agent_type:-customer_support}"

    echo -e "\nChatting with ${BOLD}${agent_type}${NC} agent. Type 'quit' to exit.\n"

    local conv_id=""

    while true; do
        read -p "You: " user_input
        [ "$user_input" = "quit" ] && break
        [ -z "$user_input" ] && continue

        response=$(call_agent "$agent_type" "$user_input" "$conv_id")
        conv_id=$(get_conversation_id "$response")

        echo -e "\n${BOLD}Agent:${NC}"
        pretty_response "$response"
        echo ""
    done

    echo -e "\nGoodbye!"
}

# ==============================================================================
# Main
# ==============================================================================

case "${1:-help}" in
    setup)  cmd_setup ;;
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    demo)   cmd_demo ;;
    chat)   cmd_chat ;;
    *)
        echo "Usage: $0 {setup|start|stop|status|demo|chat}"
        echo ""
        echo "  setup   - First-time setup: build, start, pull model"
        echo "  start   - Start services (model already pulled)"
        echo "  stop    - Stop all services"
        echo "  status  - Check service health"
        echo "  demo    - Run automated demo of all 3 agents"
        echo "  chat    - Interactive chat with an agent"
        ;;
esac
