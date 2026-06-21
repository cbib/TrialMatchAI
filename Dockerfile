FROM --platform=linux/amd64 python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts

RUN uv sync --frozen --no-dev --extra llm --extra gpu --extra entity

VOLUME ["/app/data", "/app/models", "/app/results"]

CMD ["trialmatchai-healthcheck"]
