FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION}
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project metadata and source
COPY pyproject.toml README.md ./
COPY app ./app

# Install dependencies and the project itself (for entrypoint script)
RUN poetry install --no-interaction --no-ansi

EXPOSE 7860

# Default Gradio binding for containers; can be overridden in compose
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

CMD ["poetry", "run", "chat-with-llm"]
