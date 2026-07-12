
# ============================================================
# Estágio de construção
# ============================================================

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala primeiro somente as dependências para aproveitar o cache.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# Copia o código e instala o próprio projeto.
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-editable


# ============================================================
# Estágio final de execução
# ============================================================

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ARG LOCAL_UID=1000
ARG LOCAL_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    UV_NO_SYNC=1 \
    UV_NO_ENV_FILE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HOME=/home/appuser \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${LOCAL_GID}" appgroup \
    && useradd \
        --uid "${LOCAL_UID}" \
        --gid "${LOCAL_GID}" \
        --create-home \
        --shell /bin/bash \
        appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

RUN mkdir -p \
        /app/data/raw \
        /app/data/interim \
        /app/data/processed \
        /app/data/features \
        /app/models/checkpoints \
        /app/models/registry \
        /app/models/exported \
        /app/reports/metrics \
        /app/reports/figures \
        /app/reports/predictions \
        /app/.dvc/cache \
        /tmp/torchinductor \
        /tmp/.cache \
    && chown -R appuser:appgroup \
        /app \
        /opt/venv \
        /home/appuser \
        /tmp/torchinductor \
        /tmp/.cache

USER appuser

CMD ["uv", "run", "dvc", "repro"]
