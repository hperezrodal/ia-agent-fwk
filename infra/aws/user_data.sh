#!/bin/bash
# ─────────────────────────────────────────────────────────────
# EC2 User Data — Installs Docker, NVIDIA Container Toolkit,
# and Ollama with GPU support on Deep Learning AMI (Ubuntu 22.04)
# ─────────────────────────────────────────────────────────────
set -euo pipefail
exec > >(tee /var/log/user-data.log) 2>&1

echo "=== Starting LLM server setup $(date) ==="

# ── 1. System updates ────────────────────────────────────────
apt-get update -y
apt-get upgrade -y

# ── 2. Install Docker ────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu
    systemctl enable docker
    systemctl start docker
fi

# ── 3. NVIDIA Container Toolkit ──────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# ── 4. Install Docker Compose ────────────────────────────────
if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
    apt-get install -y docker-compose-plugin
fi

# ── 5. Install Ollama ────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Configure Ollama to listen on all interfaces
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<'OVERRIDE'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
OVERRIDE

systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is ready"
        break
    fi
    sleep 2
done

# ── 6. Pull models ───────────────────────────────────────────
MODELS="${ollama_models}"
for model in $MODELS; do
    echo "Pulling model: $model"
    ollama pull "$model" || echo "WARNING: Failed to pull $model"
done

# ── 7. Verify GPU access ─────────────────────────────────────
echo "=== GPU Status ==="
nvidia-smi || echo "WARNING: nvidia-smi not available"

echo "=== Ollama models ==="
ollama list || echo "WARNING: ollama list failed"

echo "=== Setup complete $(date) ==="
