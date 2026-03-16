.PHONY: setup lint typecheck test check serve bench-smoke

setup:
	uv sync --dev

lint:
	uv run ruff check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check: lint typecheck test

serve:
	uv run python -m uvicorn switchyard.gateway:create_app --factory --host 127.0.0.1 --port 8000

bench-smoke:
	uv run python -m switchyard.bench.cli --request-count 3
