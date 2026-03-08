FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and install Python + dependencies
COPY .python-version pyproject.toml uv.lock ./
RUN uv python install && uv sync --frozen --no-dev

# Copy application code
COPY main.py ./
COPY app/ ./app/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
