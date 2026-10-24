.PHONY: sync test test-slow lint fmt requirements smoke analyze mlflow mlflow-ui dashboard \
        modal-check modal-pilot modal-grid modal-apps modal-log modal-stop modal-fetch \
        check-runs keep-runs api-smoke

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

# Cloud GPU job on Modal (plan.md §2.1). The long commands live here, so the runbook only has
# short lines: a long line copied out of a terminal can come back split in two.
MODAL = uv run --with modal==1.6.2 modal
MODAL_RUN = caffeinate -is $(MODAL) run --detach scripts/modal_eval.py
CODE = src configs scripts pyproject.toml requirements-cuda.txt

modal-check:     ## before a Modal job: code committed, Modal token valid
	@test -z "$$(git status --porcelain -- $(CODE))" || \
	  { git status --short -- $(CODE); echo "commit these first: every run records the commit"; exit 1; }
	$(MODAL) token info

modal-pilot:     ## Modal: local_ll once (builds the image; counts as rerun 1)
	$(MODAL_RUN) --reruns 1 --configs configs/eval/local_ll.yaml

modal-grid:      ## Modal: both local configs up to 3 reruns; skips finished runs (LLM_EVAL_GPU=L4 for an L4)
	$(MODAL_RUN)

modal-apps:      ## list Modal apps; a job still running shows as "ephemeral (detached)"
	$(MODAL) app list

modal-log:       ## save a job's full log (Starter keeps logs one day): make modal-log APP=ap-...
	$(MODAL) app logs $(APP) --since 1d --timestamps > data/processed/modal_job.log

modal-stop:      ## stop a running job: make modal-stop APP=ap-...
	$(MODAL) app stop $(APP)

modal-fetch:     ## copy the whole runs volume into data/processed/modal/ (manual fallback)
	mkdir -p data/processed/modal
	$(MODAL) volume get llm-eval-runs / data/processed/modal

RUNS ?= data/processed/runs/local_ll data/processed/runs/local_gen
KEEP ?= 3
check-runs:      ## check finished runs in RUNS (default: the cloud grid)
	uv run python scripts/check_runs.py $(RUNS) --keep $(KEEP)

keep-runs:       ## copy KEEP complete runs per config from RUNS into data/samples/runs
	uv run python scripts/check_runs.py $(RUNS) --keep $(KEEP) --copy-to data/samples/runs

api-smoke:       ## 24 API requests (under $0.01): keys, adapters and reply format
	uv run llm-eval run --config configs/eval/api_smoke.yaml
	uv run python scripts/check_runs.py data/processed/runs/api_smoke --keep 1
