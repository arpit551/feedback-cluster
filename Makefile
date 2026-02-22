.PHONY: install install-dev run test test-cov lint format clean

install:
	python -m venv .venv
	.venv/bin/pip install -e .

install-dev:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

run:
	.venv/bin/uvicorn cluster_api.app:app --reload

test:
	.venv/bin/pytest

test-cov:
	.venv/bin/pytest --cov=cluster_api --cov-report=term-missing

lint:
	.venv/bin/ruff check src/ tests/

format:
	.venv/bin/ruff format src/ tests/

clean:
	rm -rf .venv .pytest_cache .ruff_cache .coverage *.db dist build src/*.egg-info
