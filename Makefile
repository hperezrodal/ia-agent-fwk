# ==============================================================================
# ia-agent-fwk -- Developer Convenience Commands
# ==============================================================================
# Usage:  make <target>
#
# Run `make help` (or just `make`) to see all available targets.
# ==============================================================================

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-unit test-integration test-e2e test-coverage docker-up docker-down docker-reset docker-build docker-prod-up docker-prod-down worker beat ci clean

help: ## Show this help message
	@echo "ia-agent-fwk development commands"
	@echo ""
	@echo "Usage:  make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

# ------------------------------------------------------------------------------
# Code Quality
# ------------------------------------------------------------------------------

lint: ## Run ruff linter on source and tests
	ruff check src/ tests/

format: ## Auto-format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run mypy strict type checking on source
	mypy src/

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

test: test-unit ## Run unit tests (alias for test-unit)

test-unit: ## Run unit tests
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires Docker services)
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests (requires full system)
	pytest tests/e2e/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/unit/ --cov=src/ia_agent_fwk --cov-report=term-missing --cov-fail-under=80

# ------------------------------------------------------------------------------
# Docker Infrastructure
# ------------------------------------------------------------------------------

docker-up: ## Start development infrastructure (PostgreSQL, Redis, Qdrant)
	docker compose up -d

docker-down: ## Stop development infrastructure (preserve data)
	docker compose down

docker-reset: ## Stop infrastructure and remove all data volumes
	docker compose down -v

docker-build: ## Build the production Docker image
	docker build -f docker/Dockerfile -t ia-agent-fwk:latest .

docker-prod-up: ## Start the full production stack
	docker compose -f docker/docker-compose.prod.yml up -d --build

docker-prod-down: ## Stop the full production stack
	docker compose -f docker/docker-compose.prod.yml down

worker: ## Start a Celery worker locally (development)
	celery -A ia_agent_fwk.execution.celery_app worker --loglevel=info --pool=prefork --concurrency=4

beat: ## Start Celery Beat locally (development)
	celery -A ia_agent_fwk.execution.celery_app beat --loglevel=info

# ------------------------------------------------------------------------------
# Composite Targets
# ------------------------------------------------------------------------------

ci: lint typecheck test-unit ## Run full CI pipeline (lint + typecheck + test)

clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
