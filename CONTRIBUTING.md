# Contributing to ia-agent-fwk

Thank you for your interest in contributing to ia-agent-fwk! This document explains how to get started, the standards we follow, and the process for submitting changes.

## Getting Started

### Prerequisites

- Python 3.11 or later
- Docker and Docker Compose (for integration tests)
- Git

### Development Setup

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/<your-username>/ia-agent-fwk.git
   cd ia-agent-fwk
   ```

2. **Create a virtual environment** and install in editable mode:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   make install          # pip install -e ".[dev]"
   ```

3. **Start infrastructure** (PostgreSQL, Redis, Qdrant) if you need integration tests:

   ```bash
   make docker-up
   ```

4. **Verify everything works**:

   ```bash
   make ci               # runs lint + typecheck + unit tests
   ```

### Optional Dependencies

Install extras as needed for the modules you are working on:

```bash
pip install -e ".[huggingface]"   # HuggingFace provider
pip install -e ".[rag]"           # PDF, HTML, and Markdown loaders
pip install -e ".[weaviate]"      # Weaviate memory backend and retriever
pip install -e ".[slack]"         # Slack integration
pip install -e ".[email]"         # Email integration
```

## Code Style

This project enforces strict code quality standards through automated tooling.

### Linting and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) with nearly all rules enabled:

```bash
make lint              # check for lint errors
make format            # auto-format and auto-fix
```

### Type Checking

All source code must pass [mypy](https://mypy-lang.org/) in strict mode:

```bash
make typecheck         # mypy --strict on src/
```

### Code Conventions

- Add `from __future__ import annotations` at the top of every Python file.
- Use `TYPE_CHECKING` guards for type-only imports.
- Use `ConfigDict(frozen=True)` for Pydantic v2 models.
- Follow the ABC pattern for extensible base classes (see `LLMProvider`, `Agent`, `Tool`, `MemoryBackend`).
- Use the Factory + Registry pattern with lazy colon-delimited dotted-path imports for provider registration.
- Define module-specific exception hierarchies (e.g., `LLMProviderError`, `AgentError`).

## Testing

### Running Tests

```bash
make test              # unit tests only (fast, no external deps)
make test-integration  # integration tests (requires Docker services)
make test-coverage     # unit tests with coverage report
```

### Writing Tests

- Place unit tests in `tests/unit/test_<module>/`.
- Place integration tests in `tests/integration/`.
- Mark tests with `@pytest.mark.unit` or `@pytest.mark.integration`.
- pytest-asyncio is configured with `mode=auto`, so you do not need `@pytest.mark.asyncio` on async tests.
- Aim for meaningful coverage. All new features should include tests that cover both happy paths and error cases.

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/<short-description>` for new features
- `fix/<short-description>` for bug fixes
- `docs/<short-description>` for documentation changes
- `refactor/<short-description>` for refactoring

### Commit Messages

Write clear, concise commit messages. Use present tense (e.g., "Add retry logic to LLM provider" not "Added retry logic").

### Pull Request Process

1. **Create a branch** from `main`.
2. **Make your changes** following the code style and testing guidelines above.
3. **Run the full CI check** locally before pushing:

   ```bash
   make ci
   ```

4. **Push your branch** and open a pull request against `main`.
5. **Fill out the PR template** completely, including the checklist.
6. A maintainer will review your PR. Please respond to feedback promptly.

### What We Look For in Reviews

- Code follows project conventions (linting, typing, patterns).
- Tests are included and meaningful.
- No unnecessary changes to unrelated files.
- CHANGELOG.md is updated for user-facing changes.

## Reporting Issues

- Use the **Bug Report** template for bugs.
- Use the **Feature Request** template for enhancements.
- Search existing issues before opening a new one to avoid duplicates.

## License

By contributing to ia-agent-fwk, you agree that your contributions will be licensed under the [MIT License](LICENSE).
