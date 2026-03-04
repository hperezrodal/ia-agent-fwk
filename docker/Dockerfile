# ==============================================================================
# ia-agent-fwk -- Production Multi-Stage Dockerfile
# ==============================================================================
# Single image with entrypoint script that selects mode (api/worker/beat).
#
# Build:   docker build -f docker/Dockerfile -t ia-agent-fwk:latest .
# Run:     docker run -e MODE=api  ia-agent-fwk:latest
#          docker run -e MODE=worker ia-agent-fwk:latest
# ==============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (gcc for compiled packages like tiktoken)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition first for better layer caching
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

# Install the package and all dependencies
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/hperezrodal/ia-agent-fwk"
LABEL org.opencontainers.image.version="0.1.0"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and config
COPY --from=builder /build/src/ src/
COPY --from=builder /build/config/ config/

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER appuser

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]
