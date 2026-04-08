# linux/amd64 — required by AgentBeats submission
FROM --platform=linux/amd64 python:3.11-slim

# System deps for scientific code reproduction
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    libhdf5-dev \
    libopenblas-dev \
    gfortran \
    r-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "a2a-sdk[http-server]>=0.3.20" \
        "httpx>=0.28.1" \
        "pydantic>=2.11.9" \
        "python-dotenv>=1.1.1" \
        "uvicorn>=0.38.0" \
        "openai>=1.75.0" \
        "pypdf>=5.1.0"

COPY src/ ./src/

# AgentBeats requires these CLI args
ENTRYPOINT ["python", "src/server.py"]
CMD []
