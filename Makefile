.PHONY: sync test test-slow lint fmt requirements smoke analyze mlflow mlflow-ui dashboard

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
	uv export --frozen --no-dev --no-hashes --extra local --no-emit-project --no-emit-package torch -o requirements-cuda.txt
	uv export --frozen --no-dev --no-hashes --extra dashboard --no-emit-project -o src/llm_eval/serving/requirements.txt

smoke:           ## Day 1 smoke run: Qwen2.5-0.5B x HellaSwag x zero_shot x 100 examples
	uv run --extra local llm-eval run --config configs/eval/smoke_qwen_hellaswag.yaml

analyze:         ## results document from the committed runs (docs/results.md + .json)
	uv run llm-eval analyze --runs-root data/samples/runs --hypotheses configs/eval/hypotheses.yaml --out docs/results.md --json docs/results.json --readme README.md

mlflow:          ## log every committed run and the results document to the local MLflow store
	uv run --extra tracking llm-eval track --runs-root data/samples/runs --analysis docs/results.md

MLFLOW_PORT ?= 5000
mlflow-ui:       ## browse the MLflow store (1 worker, no job runner: ~0.4 GB instead of ~2.4 GB)
	@echo "runs table: http://127.0.0.1:$(MLFLOW_PORT)/#/experiments/1/runs?workflowType=machine_learning"
	MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false MLFLOW_DISABLE_AGENT_HINT=1 \
	  uv run --extra tracking mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1 --port $(MLFLOW_PORT)

dashboard:       ## Streamlit dashboard over data/samples/runs at http://localhost:8501
	uv run --extra dashboard streamlit run src/llm_eval/serving/app.py
