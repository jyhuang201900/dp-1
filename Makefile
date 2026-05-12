# Convenience targets for the dp-1 / forestseg project.
# Run `make help` for a quick reference.

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
SRC_DIR := src/forestseg
TEST_DIR := tests

.DEFAULT_GOAL := help

.PHONY: help install install-dev install-cpu install-gpu lint format format-check typecheck test test-fast clean precommit ci

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install runtime dependencies only
	$(PIP) install -e .

install-dev: ## Install with dev + test + torch (CPU) + rl extras
	$(PIP) install -e ".[dev,torch,rl]" --extra-index-url https://download.pytorch.org/whl/cpu

install-cpu: ## Install runtime + torch CPU wheels
	$(PIP) install -e ".[torch,rl]" --extra-index-url https://download.pytorch.org/whl/cpu

install-gpu: ## Install runtime + torch CUDA 12.1 wheels
	$(PIP) install -e ".[torch,rl]" --extra-index-url https://download.pytorch.org/whl/cu121

lint: ## Run ruff lint
	$(RUFF) check .

format: ## Apply ruff format + fix lint
	$(RUFF) check --fix .
	$(RUFF) format .

format-check: ## Verify formatting without modifying files
	$(RUFF) format --check .

typecheck: ## Run mypy (advisory)
	$(MYPY) $(SRC_DIR)

test: ## Run the full test suite
	$(PYTEST)

test-fast: ## Run tests excluding slow markers
	$(PYTEST) -m "not slow"

precommit: ## Run pre-commit on all files
	pre-commit run --all-files

ci: lint format-check test ## Run lint + format-check + tests (mirrors CI)

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
