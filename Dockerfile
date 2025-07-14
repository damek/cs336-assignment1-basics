# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# -------- system Python & tooling ------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1    # tell uv to use /usr python
ENV UV_PYTHON=python3.11

WORKDIR /workspace        # neutral path; no per-user home dirs

# -------- dependency layer -------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --system --no-editable   \
 && chmod -R a+rX /usr/local/lib/python3.11   \
 && rm -rf ~/.cache/uv                         # keep image slim

# -------- project code -----------------------------------------------------------
COPY cs336_basics ./cs336_basics
COPY README.md ./

# Install *non-editable* so no write is needed at runtime
RUN uv pip install --system . --no-deps

# Everything in /workspace is already readable; tighten perms just in case
RUN chmod -R a+rX /workspace

# -------- runtime ----------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
EXPOSE 8888
CMD ["python3.11", "-m", "cs336_basics.train"]
