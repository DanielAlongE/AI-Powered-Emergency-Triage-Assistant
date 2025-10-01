# Multi-stage build for AI-Powered Emergency Triage Assistant
# OpenAI-based deployment (no GPU required)

FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies and poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry==2.2.1

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy dependency files (leverage Docker cache)
COPY pyproject.toml poetry.lock* ./

# Install dependencies using poetry
# Regenerate lock file if pyproject.toml changed, then install
# Note: Optional dependencies are excluded by default (saves ~2GB for OpenAI deployment)
RUN poetry install --no-root && \
    rm -rf $POETRY_CACHE_DIR

# Production image
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder (virtualenv)
COPY --from=builder /build/.venv /app/.venv

# Set PATH to use virtualenv
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY config/ ./config/

# Create directories for persistent data
RUN mkdir -p /app/data/chroma_openai /app/data/sessions

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/')" || exit 1

# Run the application using the venv's interpreter
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
