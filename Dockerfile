# CPU-only image: runs evals (slowly) and the dashboard. GPU users run natively (see README).
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/data/hf
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /bin/
WORKDIR /app

# dependencies first (cached layer), project second
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras --no-install-project
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --all-extras

# provenance for run.json when the image has no .git (see docs/design.md §7.5)
ARG LLM_EVAL_GIT_SHA=""
ENV LLM_EVAL_GIT_SHA=$LLM_EVAL_GIT_SHA
ENV PATH="/app/.venv/bin:$PATH"
VOLUME ["/data", "/app/data"]
ENTRYPOINT ["llm-eval"]
CMD ["--help"]
