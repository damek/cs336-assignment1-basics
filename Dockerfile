# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 


# Install Python 3.11 (available in Ubuntu 22.04 repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Configure uv to use system python
ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON=python3.11

WORKDIR /app

# Copy only dependency files (layer caching)
COPY pyproject.toml uv.lock ./

# Install ALL dependencies (including jupyter, pytest, etc.)
RUN uv sync --locked

# Copy source code
COPY cs336_basics ./cs336_basics

# Expose Jupyter port (doesn't hurt even if not using jupyter)
EXPOSE 8888

# Default: run training (but you can override to run jupyter or anything else)
CMD ["uv", "run", "python", "-m", "cs336_basics.train"]
