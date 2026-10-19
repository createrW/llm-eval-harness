.PHONY: sync test test-slow lint fmt requirements smoke

sync:            ## create/update .venv with every extra and the dev group
	uv sync --all-extras --all-groups

test:            ## offline unit tests
	uv run pytest

test-slow:       ## tests that download Qwen2.5-0.5B (needs network, ~1 GB)
	uv run pytest -m slow

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

requirements:    ## regenerate requirements*.txt from uv.lock (never edit by hand)
	uv export --frozen --no-dev --no-hashes --extra dashboard -o requirements.txt
	uv export --frozen --no-dev --no-hashes --extra local --extra report --emit-index-url -o requirements-local.txt

smoke:           ## Day 1 smoke run: Qwen2.5-0.5B x HellaSwag x zero_shot x 100 examples
	uv run --extra local llm-eval run --config configs/eval/smoke_qwen_hellaswag.yaml
