# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 

# Install Python 3.11 (available in Ubuntu 22.04 repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user with UID 1000 (matching RunAI's runtime user)
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g 1000 -m appuser

# Switch to non-root user for all subsequent operations
USER appuser
WORKDIR /home/appuser/app

# Configure uv to use system python
ENV UV_SYSTEM_PYTHON=1
ENV UV_PYTHON=python3.11
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy only dependency files (as appuser)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Install ALL dependencies (including jupyter, pytest, etc.)
RUN uv sync --locked

# Copy source code (as appuser)
COPY --chown=appuser:appuser cs336_basics ./cs336_basics
COPY --chown=appuser:appuser README.md ./

# Install package in editable mode (now safe because appuser owns everything)
RUN uv pip install -e . --no-deps

# Expose Jupyter port
EXPOSE 8888

# Default: run training (can use uv run safely now)
CMD ["uv", "run", "python", "-m", "cs336_basics.train"]
