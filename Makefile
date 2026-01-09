.PHONY: install dev test lint format api ui clean help

# Default target
help:
	@echo "Priqualis Development Commands"
	@echo "=============================="
	@echo "make install    - Install production dependencies"
	@echo "make dev        - Install with dev dependencies"
	@echo "make test       - Run tests with coverage"
	@echo "make lint       - Run linter (ruff)"
	@echo "make format     - Format code (ruff)"
	@echo "make typecheck  - Run mypy"
	@echo "make api        - Start FastAPI server"
	@echo "make ui         - Start Streamlit UI"
	@echo "make clean      - Remove build artifacts"
	@echo "make qdrant     - Start Qdrant via Docker"

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest

test-cov:
	pytest --cov=src/priqualis --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Code quality
lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src

# Run services
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/app.py --server.port 8501

# Infrastructure
qdrant:
	docker run -p 6333:6333 -p 6334:6334 \
		-v $(PWD)/qdrant_storage:/qdrant/storage \
		qdrant/qdrant

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete